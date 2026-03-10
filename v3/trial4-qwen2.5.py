import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import evaluate
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
).to(device)


def get_new_lora_model():

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    ).to(device)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base, config)

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    return model

def generate_synthetic_edit(model, title, context):

    prompt = f"""
You are generating a new fact.

Title: {title}
Context: {context}

Create:
Question:
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.9
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Question:" in text and "Answer:" in text:
        q = text.split("Question:")[1].split("Answer:")[0].strip()
        a = text.split("Answer:")[1].strip()

        return {"question": q, "answer": a}

    return None

def get_lora_grads(model, tokenizer, facts):
    """
    Compute LoRA gradients for each protected fact.
    Returns a matrix G where each row is a flattened gradient vector.
    """

    grads = []

    for fact in facts:
        inputs = tokenizer(fact, return_tensors="pt").to(model.device)

        model.zero_grad()

        # Forward
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Backward
        loss.backward()

        # Collect gradients of only LoRA parameters
        grad_vec = []
        for name, param in model.named_parameters():
            if "lora" in name and param.grad is not None:
                grad_vec.append(param.grad.detach().flatten())

        grad_vec = torch.cat(grad_vec)
        grads.append(grad_vec)

    G = torch.stack(grads)
    return G

def compute_null_space(G, rank=32):
    """
    Given gradient matrix G (num_facts × num_params),
    compute projection matrix that removes directions
    where G has high variance.
    """

    # PCA decomposition
    U, S, Vt = torch.pca_lowrank(G, q=min(rank, min(G.shape)))

    # Keep only dominant directions
    V = Vt  # principal directions (param_dim × q)

    # Null space projection matrix
    P_null = torch.eye(V.shape[0], device=G.device) - V @ V.T
    return P_null

def reward_edit(edit):

    answer = edit["answer"]

    reward = 0

    # length reward
    reward += min(len(answer.split()) / 10, 1)

    # logical markers
    if any(x in answer.lower() for x in
           ["because","therefore","thus","implies"]):
        reward += 0.5

    # penalize repetition
    tokens = answer.split()
    reward -= (len(tokens) - len(set(tokens))) * 0.05

    return reward

def rl_generate_best_edit(model, title, context, k=2):

    best_score = -1e9
    best_edit = None

    for _ in range(k):

        e = generate_synthetic_edit(model, title, context)

        if e:
            score = reward_edit(e)

            if score > best_score:
                best_score = score
                best_edit = e

    return best_edit

def compute_nsp_shield(model, facts):

    grad_vectors = []

    for fact in facts:

        model.zero_grad()

        inputs = tokenizer(fact, return_tensors="pt").to(device)

        loss = model(
            **inputs,
            labels=inputs["input_ids"]
        ).loss

        loss.backward()

        grads = []

        for name, p in model.named_parameters():

            if "lora" in name and p.grad is not None:
                grads.append(p.grad.view(-1))

        g_vec = torch.cat(grads)

        grad_vectors.append(g_vec)

    G = torch.stack(grad_vectors)   # [num_facts , param_dim]

    # PCA in gradient space
    U, S, V = torch.pca_lowrank(G, q=min(16, G.shape[0]))

    # V: [param_dim , q]
    d = V.shape[0]




    return V

def perform_update(model, edit, V=None):

    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    text = f"Question: {edit['question']} Answer: {edit['answer']}"

    inputs = tokenizer(text, return_tensors="pt").to(device)

    loss = model(
        **inputs,
        labels=inputs["input_ids"]
    ).loss

    loss.backward()

    if V is not None:

      with torch.no_grad():

          # collect all LoRA gradients into one vector
          grads = []
          shapes = []

          for n,p in model.named_parameters():
              if "lora" in n and p.grad is not None:
                  grads.append(p.grad.view(-1))
                  shapes.append(p.grad.shape)

          g_vec = torch.cat(grads)

          # project gradient into null space
          g_proj = g_vec - V @ (V.T @ g_vec)

          # write projected gradients back
          idx = 0
          for n,p in model.named_parameters():
              if "lora" in n and p.grad is not None:

                  size = p.grad.numel()

                  p.grad = g_proj[idx:idx+size].view_as(p.grad)

                  idx += size

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

def measure_forgetting(model, facts):

    scores = []

    for fact in facts:

        inputs = tokenizer(fact, return_tensors="pt").to(device)

        with torch.no_grad():

            loss = model(
                **inputs,
                labels=inputs["input_ids"]
            ).loss

        scores.append(torch.exp(-loss).item())

    return np.mean(scores)

def generate_answer(model, tokenizer, question, context):

    prompt = f"""
Context: {context}

Question: {question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
    else:
        answer = text.strip()

    return answer.split("\n")[0]

from evaluate import load

squad_metric = load("squad")

def evaluate_squad(model, tokenizer, dataset):

    predictions = []
    references = []

    for item in tqdm(dataset, desc="Evaluating SQuAD"):

        question = item["question"]
        context = item["context"]

        prompt = f"Question: {question}\nContext: {context}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append({
            "id": item["id"],
            "prediction_text": pred
        })

        references.append({
            "id": item["id"],
            "answers": item["answers"]
        })

    results = squad_metric.compute(
        predictions=predictions,
        references=references
    )

    return results["f1"], results["exact_match"]

import numpy as np

metrics = {
    "step": [],
    "method": [],
    "retention": [],
    "forgetting": [],
    "squad_f1": [],
    "squad_em": [],
    "drift": []
}

import torch
import torch.nn.functional as F

def measure_drift(model_a, model_b, text="The capital of France is"):

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out_a = model_a(**inputs).logits
        out_b = model_b(**inputs).logits

    a = out_a.view(-1)
    b = out_b.view(-1)

    cos = F.cosine_similarity(a, b, dim=0)

    return 1 - cos.item()

def compute_forgetting(retention):
    return 1 - retention

squad_dataset = load_dataset("squad", split="validation[:30]")
def run_analysis():

    reference_knowledge = [
      "The capital of France is Paris.",
      "The capital of Germany is Berlin.",
      "The capital of Italy is Rome.",
      "Water freezes at 0 degrees Celsius.",
      "Water boils at 100 degrees Celsius.",
      "Gravity pulls objects toward Earth.",
      "The Earth orbits the Sun.",
      "Humans breathe oxygen.",
      "The Pacific Ocean is the largest ocean.",
      "The speed of light is about 3e8 m/s."
    ]

    new_passage = {
        "title": "Apollo 11",
        "context": "Jerome Wiesner opposed the flight due to safety concerns."
    }

    model_nsp = get_new_lora_model()
    model_std = get_new_lora_model()

    base_model = get_new_lora_model()

    print("Computing NSP shield...")
    P_null = compute_nsp_shield(model_nsp, reference_knowledge)

    for i in range(5):

        print("first edit start")

        print("generating edit...")
        edit = rl_generate_best_edit(
            model_nsp,
            new_passage["title"],
            new_passage["context"]
        )

        print("edit generated")

        print("updating NSP model...")
        perform_update(model_nsp, edit, P_null)

        print("NSP update done")

        print("updating standard model...")
        perform_update(model_std, edit, None)

        print("standard update done")

        print("first edit end")

        r_nsp = measure_forgetting(model_nsp, reference_knowledge)
        r_std = measure_forgetting(model_std, reference_knowledge)

        f_nsp = compute_forgetting(r_nsp)
        f_std = compute_forgetting(r_std)

        drift_nsp = measure_drift(base_model, model_nsp)
        drift_std = measure_drift(base_model, model_std)

        print("Edit generated")

        metrics["step"] += [i, i]
        metrics["method"] += ["NSP", "Standard"]

        metrics["retention"] += [r_nsp, r_std]
        metrics["forgetting"] += [f_nsp, f_std]
        metrics["drift"] += [drift_nsp, drift_std]

    print("Evaluating SQuAD")

    f1_nsp, em_nsp = evaluate_squad(model_nsp, tokenizer, squad_dataset)
    f1_std, em_std = evaluate_squad(model_std, tokenizer, squad_dataset)

    metrics["squad_f1"] += [f1_nsp, f1_std]
    metrics["squad_em"] += [em_nsp, em_std]
    metrics["squad_f1"] += [None]*(len(metrics["step"]) - len(metrics["squad_f1"]))
    metrics["squad_em"] += [None]*(len(metrics["step"]) - len(metrics["squad_em"]))

    visualize_results(metrics)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def metrics_to_df(metrics):
    return pd.DataFrame(metrics)

def plot_retention(df):

    sns.lineplot(
        data=df,
        x="step",
        y="retention",
        hue="method",
        marker="o"
    )

    plt.title("Retention vs Edit Step")
    plt.show()

def plot_forgetting(df):

    sns.lineplot(
        data=df,
        x="step",
        y="forgetting",
        hue="method",
        marker="o"
    )

    plt.title("Forgetting vs Edit Step")
    plt.show()

def plot_drift(df):

    sns.lineplot(
        data=df,
        x="step",
        y="drift",
        hue="method",
        marker="o"
    )

    plt.title("Model Drift vs Edits")
    plt.show()

def plot_tradeoff(df):

    sns.scatterplot(
        data=df,
        x="retention",
        y="forgetting",
        hue="method",
        s=100
    )

    plt.title("Retention vs Forgetting Tradeoff")
    plt.show()

import numpy as np

def radar_chart(df):

    categories = ["retention", "forgetting", "drift"]

    methods = df["method"].unique()

    values = []

    for m in methods:
        subset = df[df["method"] == m]
        vals = [subset[c].mean() for c in categories]
        values.append(vals)

    values = np.array(values)

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    for i, m in enumerate(methods):

        stats = np.concatenate((values[i], [values[i][0]]))
        ang = np.concatenate((angles, [angles[0]]))

        ax.plot(ang, stats, label=m)
        ax.fill(ang, stats, alpha=0.2)

    ax.set_thetagrids(angles * 180/np.pi, categories)

    plt.legend()
    plt.title("Editing Method Comparison")
    plt.show()

def visualize_results(metrics):

    df = metrics_to_df(metrics)

    plot_retention(df)
    plot_forgetting(df)
    plot_drift(df)
    plot_tradeoff(df)
    radar_chart(df)

run_analysis()