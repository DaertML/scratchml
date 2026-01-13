# Fix Plan - AI library from scratch

Prioritized task list for implementing the AI library from scratch. The AI agent should:
1. Study this list
2. Choose the MOST IMPORTANT item
3. Search the codebase to verify it's not implemented
4. Implement it fully
5. Test it
6. Remove it from this list
7. Commit changes

---

## Priority Items (High)

1. Model wrapper of gpt2 using existing layers and activations:
- [X] Model wrapper script in models/:
 - Implement gpt2 model architecture architecture. DO NOT use huggingface or dependencies, only the existing layers and activations in the project. Connect the needed layers into a class, to simplify the training and the inference scripts.

2. Model wrapper of llama3 using existing layers and activations:
- [X] Model wrapper script in models/:
 - Implement llama3 model architecture architecture. DO NOT use huggingface or dependencies, only the existing layers and activations in the project. Connect the needed layers into a class, to simplify the training and the inference scripts.

---

## Priority Items (Medium)

1. Training script of gpt2 using existing layers and activations:
- [ ] Training script:
 - Implement gpt2 transformer architecture training with dummy data. DO NOT use huggingface or dependencies, only the existing layers and activations in the project
 - Fix the existing train_gpt2.py. Execute the code for every code change, make fixes and changes until it works properly.
---
