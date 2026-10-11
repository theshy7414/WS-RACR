"""WS-RACR: world-state-centered, retrieval-augmented human-robot collaboration.

Package map (system layers -> thesis chapters):

========================  ==============================================  =======
Sub-package               Role                                            Thesis
========================  ==============================================  =======
``perception``            YOLOv8 / MediaPipe / TSM, RealSense, localisation  Ch. 3
``world_state``           W_t = {S_t, O_t, H_t, R_t}, stabilisation          Ch. 3
``memory``                interaction log, temporal log, SOP, KG memory      Ch. 4
``database``              PostgreSQL + pgvector (Plan RAG), embeddings       Ch. 4
``decision``              Layer 1/2/3 policy, deictic grounding, parsing     Ch. 4
``orchestrator``          HRCOrchestrator: planning / execution stages       Ch. 4
``execution``             skills, backends (TMScript / MoveIt2), safety      Ch. 3-4
``hri``                   speech-to-text, text-to-speech, spoken queries     Ch. 3
``app``                   ``HRCSystem`` runtime and command-line entry       Ch. 3
========================  ==============================================  =======

"Layer 1/2/3" always means the three collaborative *decision* layers
(instruction understanding / execution feasibility / execution risk); the three
*system* layers are perception, reasoning/memory and execution.
"""

__version__ = "1.0.0"
