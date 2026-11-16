"""68 of MediaPipe Face Mesh's 478 landmarks, in the iBUG 68-point order the FLAME tracker expects (#126).

The tracker fits FLAME to a portrait through 68 points (jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, lips 48-67)
and an embedding that says where each lies on FLAME's surface. Both now come from MediaPipe (Apache-2.0): the
front end detects these points, and `measure.py` measures where the same points fall on FLAME. So what matters is
that the two agree, which they do by construction, not that each point matches iBUG's definition exactly.

"Right" is the subject's right, on the image's left. Every left index mirrors its right partner in MediaPipe's
symmetric mesh. Plain constants, importable from LAM's environment and from MediaPipe's.
"""

JAW = (234, 93, 132, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 361, 323, 454)
RIGHT_BROW = (70, 63, 105, 66, 107)  # outer to inner
LEFT_BROW = (336, 296, 334, 293, 300)  # inner to outer
NOSE_BRIDGE = (168, 197, 5, 4)  # between the eyes down to the tip
NOSE_BASE = (98, 97, 2, 326, 327)  # right ala, through the base, to the left
RIGHT_EYE = (33, 160, 158, 133, 153, 144)  # outer corner, upper lid, inner corner, lower lid
LEFT_EYE = (362, 385, 387, 263, 373, 380)  # inner corner, upper lid, outer corner, lower lid
OUTER_LIPS = (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91)  # right corner, upper, left corner, lower
INNER_LIPS = (78, 81, 13, 311, 308, 402, 14, 178)

MEDIAPIPE_68 = JAW + RIGHT_BROW + LEFT_BROW + NOSE_BRIDGE + NOSE_BASE + RIGHT_EYE + LEFT_EYE + OUTER_LIPS + INNER_LIPS
# The irises' centres (MediaPipe's refined points), for the tracker's two pupil landmarks (68, 69).
RIGHT_IRIS, LEFT_IRIS = 468, 473

assert len(MEDIAPIPE_68) == 68 and len(set(MEDIAPIPE_68)) == 68
