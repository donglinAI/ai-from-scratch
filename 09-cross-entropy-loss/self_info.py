
import torch 
import matplotlib.pyplot as  plt 

p = torch.linspace(0.001, 1, 500)
I = -torch.log(p)
print(f"前10个值 p: {p[:10]}\n I: {I[:10]}")

plt.figure(figsize=(6,4), dpi=160)
plt.plot(p.numpy(), I.numpy(), color="blue", linewidth=2)
plt.xlabel("p")
plt.ylabel("I")
plt.title("Information vs Probability")
plt.tight_layout()
plt.savefig("self_information.png")