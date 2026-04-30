# Wheelchair Path — 3D Home Demo

Single-page Three.js demo for wheelchair navigation inside a multi-room home layout.

Features include room/door graph routing, grid stitching, a main orbit camera plus a wheelchair camera, and a camera-ray obstacle sensing stub that can trigger replanning hooks.

---

## Quick Start

```bash
git clone https://github.com/rezabahadori76/Robot.git
cd Robot
python3 -m http.server 8080
# Open: http://localhost:8080/index.html
```

---

## Project Files

| Path | Purpose |
|------|---------|
| `index.html` | Main app: scene, routing, wheelchair motion, HUD, sensor logic |
| `three.min.js` | Three.js runtime |
| `MTLLoader.js`, `OBJLoader.js`, `TDSLoader.js` | Asset loaders |
| `assets/models/...` | 3D models and textures |

---

## Core Capabilities

- Multi-room home scene with walls, doors, floor, and lighting
- Route planning via room/door graph + local grid fallback
- Dual camera split view (main + wheelchair FPV)
- Wheelchair camera obstacle sensing using NDC ray samples
- Hook points for stop/replan behavior based on nearest hits and forward clearance

---

## Runtime API (`window.ROBOT_HOUSE`)

Key objects exposed in browser console:

- `rooms`, `doors`, `bounds`, `footprint`, `nav`
- `wheelchairCameraSensor`
- `dynamicReplan` utilities

`wheelchairCameraSensor` provides:

- `scan()`
- `lastScan`
- `setMaxRangeM()`, `setMinAlertDistanceM()`
- `setNdcSamples()`
- `onNearObstacle`
- `resetAlertLatch()`

---

## Example Hook

```js
ROBOT_HOUSE.wheelchairCameraSensor.setMinAlertDistanceM(0.75);

ROBOT_HOUSE.wheelchairCameraSensor.onNearObstacle = (scan) => {
  const d = scan.forwardClearanceM ?? scan.nearest?.distance;
  if (d == null) return;
  console.warn("[sensor] near obstacle", d, "m", scan.nearest?.root);
  // stop motion, mark blocked cells, compute replan, then reset latch
};
```

---

## Notes

- This sensor is geometry-based (not semantic classification by default).
- Delayed-loading meshes are not seen by scans until they are in scene.
- Verify model licenses in `assets/models/` before redistribution.

