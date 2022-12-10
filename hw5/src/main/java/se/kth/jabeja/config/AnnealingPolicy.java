package se.kth.jabeja.config;

public enum AnnealingPolicy {
    /**
     * Linear annealing policy, temperature cools linearly
     */
    LINEAR("LINEAR"),
    /**
     * Linear annealing policy, temperature cools exponentially
     */
    EXPONENTIAL("EXPONENTIAL");

    String name;

    AnnealingPolicy(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}
