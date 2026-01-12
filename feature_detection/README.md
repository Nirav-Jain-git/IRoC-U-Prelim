
Validation module for seed-based visual matching.

This code performs robust validation of detected seed images using a robust and fast approach.
Early layers (first 7) of a pretrained MobileNetV2 are used
to extract appearance features that preserve spatial structure and surface texture.
These features are summarized and compared using cosine similarity to assess
visual consistency between the seed and candidate.

Local Binary Patterns (LBP) provide a fast and compact texture comparison, while
SIFT acts as a weak geometric check to reject clear mismatches. ORB is included
as an optional high-precision cue for confirming near-identical matches. All cues
are combined into a fusion score, which determines the final validation decision.


<img width="1196" height="512" alt="image" src="https://github.com/user-attachments/assets/7645ad8e-ba6a-41d4-8865-79c4df1781a3" />



