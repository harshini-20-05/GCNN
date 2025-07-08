# GCNNs from Scratch

This project demonstrates a **from-scratch implementation of Group Equivariant Convolutional Neural Networks (GCNNs)**, focused on achieving **rotation invariance** using cyclic groups (specifically the C4 group – 90° rotational symmetries).

---

## 📌 What We Did

- Implemented the core GCNN architecture **from scratch inside a Jupyter Notebook** (`gcnn-dl-assignment.ipynb`) without using any GCNN libraries.
- Used manually constructed rotation matrices to handle image transformations and weight sharing across rotated filters.
- Compared GCNN performance on:
  - Original upright MNIST digits.
  - The same digits rotated by 90°, 180°, and 270°.
- Verified that GCNNs maintain consistent accuracy across rotated inputs — highlighting **rotation equivariance**.

---

## 💡 Model Design

To keep the implementation **lightweight and fast**, we intentionally used a **small number of layers** in our architecture. This helps reduce model complexity and training time, which is especially useful when:

- Testing new ideas quickly.
- Running on limited hardware.
- Focusing on verifying theoretical properties like group equivariance.

> 🔸 **Note:** We used a simple CNN as the base model for this GCNN implementation.  
> 🔸 For improved accuracy or more complex datasets, deeper or modern architectures can be adopted — see **Future Scope** below.

---

## 🧪 Experiment Summary

- **Dataset Used:** MNIST (handwritten digits).
- **Model:** Custom-built GCNN using 90° rotation-aware filters.
- **Evaluation:**
  - Trained only on original (non-rotated) images.
  - Tested on rotated images to validate generalization.

### ✅ Observations:

- Standard CNNs struggle with rotated digits.
- Our GCNN achieved nearly **identical accuracy** on rotated and non-rotated test sets.
- Demonstrates that **GCNNs learn features invariant to input rotations**, making them ideal for symmetry-aware tasks.

---

## 📓 Notebook Structure

- `gcnn-dl-assignment.ipynb`
  - `Section 1:` Data loading and visualization
  - `Section 2:` GCNN layer implementation from scratch
  - `Section 3:` Training the model
  - `Section 4:` Testing with rotated digits
  - `Section 5:` Results and comparison

---

## 🧠 Key Concepts Demonstrated

- **Group equivariance** using cyclic rotations (C4 group).
- Custom convolution layers with weight tying over group transformations.
- Testing with rotation-augmented datasets to verify model robustness.
- No reliance on GCNN libraries – built using core PyTorch + NumPy only.

---

## 🚀 Run the Notebook

1. Clone this repository or download the `.ipynb` file.
2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
