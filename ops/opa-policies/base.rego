package qbitel.policy

default allow = false

# Example: only allow PQC-hybrid mode in production
allow {
  input.mode == "pqc-hybrid"
}
