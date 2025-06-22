import random
import matplotlib.pyplot as plt
from collections import deque, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cargar modelo TinyLlama
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Parámetros del experimento
AGENTS = ["agent_1", "agent_2", "agent_3", "agent_4"]
NAME_POOL = ["M", "Q"]
ROUNDS = 30
MEMORY_SIZE = 3
COMMITTED_AGENTS = {"agent_4": "Q"}  # agente comprometido siempre dice "Q"

# Estado de cada agente
agent_state = {
    name: {
        "memory": deque(maxlen=MEMORY_SIZE),
        "score": 0,
        "history": []
    }
    for name in AGENTS
}

# Formatea el historial de interacciones
def format_memory(memory):
    if not memory:
        return "No previous interactions.\n"
    return "\n".join(
        f"Round {i+1}: I chose {x['self']}, partner chose {x['partner']}, success: {x['success']}, my score: {x['score']}"
        for i, x in enumerate(memory)
    )

# Construye el prompt del agente
def build_prompt(agent_name):
    memory = format_memory(agent_state[agent_name]["memory"])
    return f"""You are an agent in a coordination game.
You and another player must choose one of the following letters: [M, Q].
If you choose the same letter, you both win +100 points.
If you choose differently, both lose -50 points.
Your goal is to maximize your score based on past results.

{memory}

Choose your next move. Answer with a single letter from the pool above."""

# Genera respuesta con TinyLlama
def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=True, temperature=0.9)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    for c in reversed(NAME_POOL):
        if c in text:
            return c
    return random.choice(NAME_POOL)  # fallback

# Interacción entre dos agentes
def interact(a1, a2, round_num):
    c1 = COMMITTED_AGENTS.get(a1) or query_model(build_prompt(a1))
    c2 = COMMITTED_AGENTS.get(a2) or query_model(build_prompt(a2))

    success = c1 == c2
    delta = 100 if success else -50

    for a, c, opp in [(a1, c1, c2), (a2, c2, c1)]:
        agent_state[a]["score"] += delta
        agent_state[a]["memory"].append({
            "self": c,
            "partner": opp,
            "success": success,
            "score": agent_state[a]["score"]
        })
        agent_state[a]["history"].append(c)

    print(f"Round {round_num + 1}: {a1}({c1}) ↔ {a2}({c2}) | {'✅' if success else '❌'}")
    return c1, c2

# Ejecuta el experimento
all_choices = []
for r in range(ROUNDS):
    a1, a2 = random.sample(AGENTS, 2)
    c1, c2 = interact(a1, a2, r)
    all_choices.append(Counter([c1, c2]))

# Visualización: evolución por ronda
m_counts, q_counts = [], []
for counter in all_choices:
    m_counts.append(counter["M"])
    q_counts.append(counter["Q"])

plt.plot(m_counts, label="M")
plt.plot(q_counts, label="Q")
plt.xlabel("Round")
plt.ylabel("Choices per round")
plt.title("Emergence of Convention (TinyLlama)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Recuento final
final = Counter()
for a in AGENTS:
    final.update(agent_state[a]["history"])

print("\n📊 Final Choice Totals:")
for c in NAME_POOL:
    print(f"{c}: {final[c]}")

# Detección de convención
dominant = final.most_common(1)[0]
if dominant[1] > sum(final.values()) * 0.75:
    print(f"\n✅ Convention emerged: {dominant[0]}")
else:
    print("\n❌ No clear convention emerged.")
