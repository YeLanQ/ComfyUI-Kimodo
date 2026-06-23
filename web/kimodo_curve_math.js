export function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

// ---- 2D monotonic Hermite (kept for backward compat) ----

export function computeMonotonicTangents(points) {
  const tangents = new Array(points.length).fill(0);

  for (let i = 1; i < points.length - 1; i++) {
    const dx = points[i + 1].x - points[i - 1].x;
    const dy = points[i + 1].y - points[i - 1].y;
    tangents[i] = dx === 0 ? 0 : dy / dx;
  }

  tangents[0] = (points[1].y - points[0].y) / (points[1].x - points[0].x);
  tangents[points.length - 1] =
    (points[points.length - 1].y - points[points.length - 2].y) /
    (points[points.length - 1].x - points[points.length - 2].x);

  return tangents;
}

export function hermiteInterp(y0, y1, m0, m1, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    (2 * t3 - 3 * t2 + 1) * y0 +
    (t3 - 2 * t2 + t) * m0 +
    (-2 * t3 + 3 * t2) * y1 +
    (t3 - t2) * m1
  );
}

export function sampleMonotonicSplineY(x, points, tangents) {
  if (points.length < 2) return 0;

  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i];
    const p1 = points[i + 1];
    if (x >= p0.x && x <= p1.x) {
      const t = (x - p0.x) / (p1.x - p0.x);
      const m0 = tangents[i] * (p1.x - p0.x);
      const m1 = tangents[i + 1] * (p1.x - p0.x);
      return clamp01(hermiteInterp(p0.y, p1.y, m0, m1, t));
    }
  }

  return x <= points[0].x ? points[0].y : points[points.length - 1].y;
}

export function getSmoothMonotonicCurveHermite(points, resolution = 100) {
  const result = [];
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const tangents = computeMonotonicTangents(sorted);

  for (let i = 0; i < resolution; i++) {
    const x = i / (resolution - 1);
    const y = sampleMonotonicSplineY(x, sorted, tangents);
    result.push({ x, y });
  }

  return result;
}

// ---- 3D chord-length parameterized Hermite ----

function dist3D(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function computeChordLengthParams(points) {
  const chords = [0];
  for (let i = 1; i < points.length; i++) {
    chords.push(chords[i - 1] + dist3D(points[i - 1], points[i]));
  }
  const total = chords[chords.length - 1];
  const t = chords.map(c => (total > 1e-8 ? c / total : c));
  return { t, total };
}

export function computeTangents3D(points, t) {
  const n = points.length;
  const tangents = [];
  for (let i = 0; i < n; i++) {
    let tx, ty, tz;
    if (i === 0) {
      tx = points[1].x - points[0].x;
      ty = points[1].y - points[0].y;
      tz = points[1].z - points[0].z;
    } else if (i === n - 1) {
      tx = points[n - 1].x - points[n - 2].x;
      ty = points[n - 1].y - points[n - 2].y;
      tz = points[n - 1].z - points[n - 2].z;
    } else {
      const dt = t[i + 1] - t[i - 1];
      if (dt > 1e-8) {
        tx = (points[i + 1].x - points[i - 1].x) / dt;
        ty = (points[i + 1].y - points[i - 1].y) / dt;
        tz = (points[i + 1].z - points[i - 1].z) / dt;
      } else {
        tx = points[i + 1].x - points[i - 1].x;
        ty = points[i + 1].y - points[i - 1].y;
        tz = points[i + 1].z - points[i - 1].z;
      }
    }
    tangents.push({ x: tx, y: ty, z: tz });
  }
  return tangents;
}

function hermiteInterpVal(v0, v1, m0, m1, t) {
  const t2 = t * t, t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * v0 +
         (t3 - 2 * t2 + t) * m0 +
         (-2 * t3 + 3 * t2) * v1 +
         (t3 - t2) * m1;
}

export function evalHermiteSegment3D(p0, p1, m0, m1, t01) {
  return {
    x: hermiteInterpVal(p0.x, p1.x, m0.x, m1.x, t01),
    y: hermiteInterpVal(p0.y, p1.y, m0.y, m1.y, t01),
    z: hermiteInterpVal(p0.z, p1.z, m0.z, m1.z, t01),
  };
}

export function sampleCurve3D(points, tangents, t, resolution = 100) {
  if (points.length < 2) return points.slice();
  const result = [];
  for (let i = 0; i < resolution; i++) {
    const u = i / (resolution - 1);
    // Find segment
    let seg = 0;
    while (seg < points.length - 2 && t[seg + 1] < u) seg++;
    const t0 = t[seg], t1 = t[seg + 1];
    const segLen = t1 - t0;
    const localT = segLen > 1e-8 ? (u - t0) / segLen : 0;
    const m0 = {
      x: tangents[seg].x * segLen,
      y: tangents[seg].y * segLen,
      z: tangents[seg].z * segLen,
    };
    const m1 = {
      x: tangents[seg + 1].x * segLen,
      y: tangents[seg + 1].y * segLen,
      z: tangents[seg + 1].z * segLen,
    };
    result.push(evalHermiteSegment3D(points[seg], points[seg + 1], m0, m1, localT));
  }
  return result;
}
