# BaddieCam BeautyCore AI for OBS

A native Windows OBS video filter designed for automatic, face-aware webcam beauty processing without a separate beauty-camera application.

## Non-negotiable design rules

- **No facial geometry or warping.** There is no eye enlargement, jaw slimming, nose shaping, or coordinate remapping.
- **No manual face positioning.** A lightweight detector and semantic face parser create the masks automatically.
- **Fail closed.** If AI confidence is poor, local beauty fades to the untouched camera instead of freezing stale masks.
- **Protected identity detail.** Eyes, lashes, brows, lips, nostrils, hair, and hard edges are protected from skin blur.
- **True zero.** Master Beauty at zero returns the original image.
- **Local inference.** Models run on the user's PC through ONNX Runtime; frames are not sent to an online service.

## Processing pipeline

1. OBS supplies unmodified webcam frames to a small asynchronous AI queue.
2. YuNet detects the most credible face.
3. A 19-class face-parsing model labels skin, brows, eyes, nose, lips, hair, and background.
4. The parser also creates automatic eye, lip, cheek, under-eye, and forehead/T-zone beauty masks.
5. A motion-aware state machine rejects implausible jumps and stabilizes the masks.
6. The latest masks are uploaded to the GPU.
7. The OBS render filter applies multi-scale skin smoothing, pore refinement, complexion evening, under-eye balancing, shine control, and glass-skin finishing only where allowed by the AI mask.

The AI mask can update at 8–15 FPS while the final OBS render remains at the scene frame rate.

## Build status

This package is complete **source code**, build scripts, effects, a verified model downloader, tests, and documentation. It is not a precompiled DLL because this environment does not contain the Windows OBS SDK, Visual Studio, or a physical camera. Build through the official OBS plug-in template and GitHub Actions, then perform live camera QA before treating it as production-ready.

On Windows, configuration verifies or downloads both pinned ONNX models by SHA-256. The overlay script also downloads them before the repository is committed, so the packaged artifact contains everything the runtime needs.


Detector note: the pinned YuNet ONNX model uses its fixed 640 x 640 input layout; BaddieCam letterboxes frames without stretching before inference.
