import { app } from "../../scripts/app.js";
import { drawCurveBackground, drawCurve } from "./kimodo_draw_utils.js";
import { getSmoothMonotonicCurveHermite } from "./kimodo_curve_math.js";

const NODE_NAME = "Kimodo_CurveToPoints";

const CURVE_COLOR = "#4af";
const POINT_COLOR = "#4af";
const HIGHLIGHT_COLOR = "#fff";
const GRID_COLOR = "#555";

const DEFAULT_CURVE = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
];

const MIN_WIDTH = 320;
const MIN_HEIGHT = 300;
const PADDING = 20;
const HEADER_HEIGHT = 30;

const EDITOR_CONFIG = {
  resolution: 100,
  activeWidth: 3,
  pointRadius: 5,
  maxPoints: 24,
  grid_x: 5,
  grid_y: 5,
};

// ---- simple PointEditor (single curve, no channel tabs) ----

class PointEditor {
  constructor(node, config, onChange) {
    this.node = node;
    this.config = config;
    this.onChange = onChange;
    this.points = [];
    this.dragIndex = null;
    this.isMouseDown = false;
    this.hoveredPointIndex = null;
    this.layout = { x: 0, y: 0, width: 0, height: 0 };
  }

  updateLayout(layout) {
    this.layout = layout;
  }

  draw(ctx) {
    if (this.layout.width <= 0 || this.layout.height <= 0) return;

    const smooth = getSmoothMonotonicCurveHermite(
      this.points,
      this.config.resolution
    );

    drawCurve(
      ctx,
      smooth,
      this.layout,
      { curveColor: CURVE_COLOR, curveWidth: this.config.activeWidth },
      this.points,
      {
        pointRadius: this.config.pointRadius,
        pointColor: POINT_COLOR,
        highlightIndex: this.hoveredPointIndex,
        highlightColor: HIGHLIGHT_COLOR,
      }
    );

    if (this.hoveredPointIndex !== null && this.points[this.hoveredPointIndex]) {
      const p = this.points[this.hoveredPointIndex];
      const px = this.layout.x + p.x * this.layout.width;
      const ctxSave = ctx;
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(px, this.layout.y);
      ctx.lineTo(px, this.layout.y + this.layout.height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
  }

  _isInBounds(mx, my) {
    const { x, y, width, height } = this.layout;
    return mx >= x && mx <= x + width && my >= y && my <= y + height;
  }

  _pointUnderCursor(cx, cy) {
    const r2 = (this.config.pointRadius * 2.5) ** 2;
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      const px = this.layout.x + p.x * this.layout.width;
      const py = this.layout.y + (1 - p.y) * this.layout.height;
      const dx = cx - px, dy = cy - py;
      if (dx * dx + dy * dy <= r2) return i;
    }
    return -1;
  }

  onMouseDown(e, localPos) {
    const [mx, my] = localPos;
    if (!this._isInBounds(mx, my)) return false;

    // Right-click to delete point
    if (e.button === 2) {
      const idx = this._pointUnderCursor(mx, my);
      if (idx !== -1) {
        const p = this.points[idx];
        if (p.x > 0.001 && p.x < 0.999 && this.points.length > 2) {
          this.points.splice(idx, 1);
          this.onChange(this.points);
        }
      }
      this.node.setDirtyCanvas(true, true);
      return true;
    }

    this.isMouseDown = true;
    this.dragIndex = this._pointUnderCursor(mx, my);

    // Add point on empty click
    if (this.dragIndex === -1 && this.points.length < this.config.maxPoints) {
      const xNorm = Math.max(0, Math.min(1, (mx - this.layout.x) / this.layout.width));
      const yNorm = Math.max(0, Math.min(1, 1 - (my - this.layout.y) / this.layout.height));
      const newPoint = { x: xNorm, y: yNorm };
      this.points.push(newPoint);
      this.points.sort((a, b) => a.x - b.x);
      this.dragIndex = this.points.indexOf(newPoint);
      this.onChange(this.points);
    }

    this.node.setDirtyCanvas(true, true);
    return true;
  }

  onMouseMove(e, localPos) {
    const [mx, my] = localPos;
    let redraw = false;

    const oldHover = this.hoveredPointIndex;
    if (this._isInBounds(mx, my)) {
      this.hoveredPointIndex = this._pointUnderCursor(mx, my);
    } else {
      this.hoveredPointIndex = null;
    }
    if (oldHover !== this.hoveredPointIndex) redraw = true;

    if (this.isMouseDown && this.dragIndex !== null) {
      if (e.buttons !== 1) return this.onMouseUp(e, localPos);
      this._dragPoint(mx, my);
      redraw = true;
    }

    if (redraw) this.node.setDirtyCanvas(true, true);
    return this._isInBounds(mx, my);
  }

  onMouseUp(e, localPos) {
    if (this.isMouseDown) {
      this.isMouseDown = false;
      this.dragIndex = null;
      this.onChange(this.points);
      return true;
    }
    return false;
  }

  onMouseLeave(e, localPos) {
    this.hoveredPointIndex = null;
    if (this.isMouseDown) this.onMouseUp(e, localPos);
    this.node.setDirtyCanvas(true, true);
    return true;
  }

  _dragPoint(mx, my) {
    if (this.dragIndex === null || !this.points[this.dragIndex]) return;
    const { x, y, width, height } = this.layout;
    const pt = this.points[this.dragIndex];
    pt.y = Math.max(0, Math.min(1, 1 - (my - y) / height));

    const isEnd = this.dragIndex === 0 || this.dragIndex === this.points.length - 1;
    if (!isEnd) {
      const xNorm = (mx - x) / width;
      const prevX = this.points[this.dragIndex - 1]?.x ?? 0;
      const nextX = this.points[this.dragIndex + 1]?.x ?? 1;
      pt.x = Math.max(prevX, Math.min(nextX, xNorm));
    }
    this.onChange(this.points);
  }

  setCurveData(pts) {
    this.points = JSON.parse(JSON.stringify(pts || []));
    this.node.setDirtyCanvas(true, true);
  }

  getCurveData() {
    return JSON.parse(JSON.stringify(this.points));
  }
}

// ---- ComfyUI extension ----

app.registerExtension({
  name: "kimodo.CurveToPoints",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== NODE_NAME) return;

    nodeType.prototype.onNodeCreated = function () {
      this.serialize_widgets = true;
      this.min_size = [MIN_WIDTH, MIN_HEIGHT];
      this.resizable = true;
      this.size = this.computeSize();

      this.curvePoints = DEFAULT_CURVE.map((p) => ({ ...p }));

      this.editor = new PointEditor(this, EDITOR_CONFIG, (pts) => {
        this.curvePoints = pts.map((p) => ({ ...p }));
        this.syncCurveToWidget();
        this.setDirtyCanvas(true, true);
      });
      this.editor.setCurveData(this.curvePoints);

      // Hidden widget to hold curve JSON for the backend
      this.jsonWidget = this.addWidget("text", "curve_json", JSON.stringify(this.curvePoints), () => {});
      this.jsonWidget.hidden = true;
      this.properties.curve_json = JSON.stringify(this.curvePoints);

      this.setDirtyCanvas(true, true);
    };

    nodeType.prototype.computeSize = function () {
      return [MIN_WIDTH, MIN_HEIGHT];
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      if (origOnConfigure) origOnConfigure.call(this, o);
      try {
        const raw = this.properties?.curve_json;
        if (raw) {
          const data = typeof raw === "string" ? JSON.parse(raw) : raw;
          if (Array.isArray(data) && data.length >= 2) {
            this.curvePoints = data.map((p) => ({ x: p.x, y: p.y }));
            this.editor.setCurveData(this.curvePoints);
          }
        }
      } catch (e) {
        console.warn("[Kimodo_CurveToPoints] Failed to parse curve_json:", e);
      }
    };

    nodeType.prototype.syncCurveToWidget = function () {
      const json = JSON.stringify(this.curvePoints);
      this.jsonWidget.value = json;
      this.properties.curve_json = json;
    };

    const origOnAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      if (origOnAdded) origOnAdded.call(this);

      // Block ComfyUI node context menu when right-click is in the curve editor area
      const graphCanvas = this.graph?.canvas;
      if (graphCanvas) {
        const origShow = graphCanvas.showNodeContextMenu;
        graphCanvas.showNodeContextMenu = function (node, ...args) {
          if (node.type === NODE_NAME && node._skipCtx) {
            delete node._skipCtx;
            return;
          }
          return origShow.call(this, node, ...args);
        };
        (this.onRemoved = this.onRemoved || []).push(() => {
          graphCanvas.showNodeContextMenu = origShow;
        });

        // Prevent native browser context menu after right-click in editor
        const canvasEl = graphCanvas.canvas;
        const handler = (e) => {
          if (this._pendingCtx && this.editor?._isInBounds(...this._pendingCtx)) {
            e.preventDefault();
          }
          this._pendingCtx = null;
        };
        canvasEl.addEventListener("contextmenu", handler);
        this.onRemoved.push(() => canvasEl.removeEventListener("contextmenu", handler));
      }

      const origOnMouseDown = this.onMouseDown;
      const origOnMouseMove = this.onMouseMove;
      const origOnMouseUp = this.onMouseUp;
      const origOnMouseLeave = this.onMouseLeave;

      this.onMouseDown = function (e, pos, canvas) {
        // Editor handles right-click for deletion and blocks ComfyUI context menu
        if (e.button === 2 && this.editor?._isInBounds(pos[0], pos[1])) {
          e.preventDefault();
          this._skipCtx = true;
          this._pendingCtx = [pos[0], pos[1]];
          if (this.editor.onMouseDown(e, pos)) {
            this.setDirtyCanvas(true, true);
            return true;
          }
          return true;
        }
        if (this.editor?.onMouseDown?.(e, pos)) {
          this.setDirtyCanvas(true, true);
          return true;
        }
        if (origOnMouseDown?.call(this, e, pos, canvas)) return true;
        return false;
      };

      this.onMouseMove = function (e, pos, canvas) {
        const handled = origOnMouseMove?.call(this, e, pos, canvas) ?? false;
        if (this.editor?.onMouseMove?.(e, pos)) return true;
        return handled;
      };

      this.onMouseUp = function (e, pos, canvas) {
        if (origOnMouseUp?.call(this, e, pos, canvas)) return true;
        this.editor?.onMouseUp?.(e, pos);
        this.syncCurveToWidget();
        return false;
      };

      this.onMouseLeave = function (e, pos, canvas) {
        if (origOnMouseLeave?.call(this, e, pos, canvas)) return true;
        const fallback = pos ?? [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
        this.editor?.onMouseUp?.(e, fallback);
        return false;
      };
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      if (this.collapsed) return;
      ctx.save();

      const area = {
        x: PADDING,
        y: PADDING + HEADER_HEIGHT,
        width: this.size[0] - 2 * PADDING,
        height: this.size[1] - PADDING - HEADER_HEIGHT - PADDING,
        grid_x: EDITOR_CONFIG.grid_x,
        grid_y: EDITOR_CONFIG.grid_y,
      };

      drawCurveBackground(ctx, area, { fillStyle: "#333", strokeStyle: GRID_COLOR });
      this.editor.updateLayout(area);
      this.editor.draw(ctx);

      // Label
      ctx.fillStyle = "#aaa";
      ctx.font = "11px Arial";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("X →", PADDING, PADDING + 6);

      ctx.textAlign = "right";
      ctx.fillText("← Z", this.size[0] - PADDING, PADDING + 6);

      ctx.restore();
    };
  },
});
