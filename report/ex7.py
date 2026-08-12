import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)


def shannon_information(p):
    return -np.log2(p)


def binary_entropy(p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def kl_bernoulli(p, q):
    eps = 1e-12
    q = np.clip(q, eps, 1 - eps)
    return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))


def fisher_bernoulli(theta):
    return 1 / (theta * (1 - theta))


def plot_shannon_information():
    p = np.linspace(0.01, 1.0, 500)
    info = shannon_information(p)

    plt.figure(figsize=(8, 5))
    plt.plot(p, info)
    plt.xlabel("Probability p")
    plt.ylabel("Information -log2(p) [bits]")
    plt.title("Shannon Information")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/shannon_information.png")
    plt.close()


def plot_binary_entropy():
    p = np.linspace(0.001, 0.999, 500)
    h = binary_entropy(p)

    plt.figure(figsize=(8, 5))
    plt.plot(p, h)
    plt.axvline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Probability p")
    plt.ylabel("Entropy H(p) [bits]")
    plt.title("Binary Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/binary_entropy.png")
    plt.close()


def plot_kl_divergence():
    true_p = 0.7
    q = np.linspace(0.01, 0.99, 500)
    kl = kl_bernoulli(true_p, q)

    plt.figure(figsize=(8, 5))
    plt.plot(q, kl)
    plt.axvline(true_p, linestyle="--", linewidth=1)
    plt.xlabel("Model parameter q")
    plt.ylabel("KL divergence D_KL(p || q) [bits]")
    plt.title("KL Divergence from Bernoulli(0.7) to Bernoulli(q)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/kl_divergence.png")
    plt.close()


def plot_fisher_information():
    theta = np.linspace(0.01, 0.99, 500)
    fisher = fisher_bernoulli(theta)

    plt.figure(figsize=(8, 5))
    plt.plot(theta, fisher)
    plt.axvline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Bernoulli parameter theta")
    plt.ylabel("Fisher information I(theta)")
    plt.title("Fisher Information for Bernoulli Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/fisher_information.png")
    plt.close()


def f(x):
    return x - x**3


def plot_morse_decomposition():
    x = np.linspace(-1.6, 1.6, 500)
    y = f(x)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, label=r"$dx/dt = x - x^3$")
    plt.axhline(0, linewidth=1)

    stable_points = [-1, 1]
    unstable_points = [0]

    for p in stable_points:
        plt.plot(p, 0, "o", markersize=10)
        plt.text(p, -0.22, f"M={p}\nstable", ha="center")

    for p in unstable_points:
        plt.plot(
            p,
            0,
            "o",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor="black",
        )
        plt.text(p, 0.18, "M=0\nunstable", ha="center")

    plt.annotate(
        "",
        xy=(-1, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", linewidth=2),
    )

    plt.annotate(
        "",
        xy=(1, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", linewidth=2),
    )

    plt.fill_between(
        x,
        -0.08,
        0.08,
        where=(x >= -1) & (x <= 1),
        alpha=0.2,
        label=r"Global attractor A = [-1, 1]",
    )

    plt.xlabel("x")
    plt.ylabel(r"$dx/dt$")
    plt.title("Morse Decomposition of dx/dt = x - x^3")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/morse_decomposition.png")
    plt.close()


def main():
    plot_shannon_information()
    plot_binary_entropy()
    plot_kl_divergence()
    plot_fisher_information()
    plot_morse_decomposition()

    print("All figures were saved in the figures folder.")


if __name__ == "__main__":
    main()
