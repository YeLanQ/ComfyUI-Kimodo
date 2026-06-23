/**
 * Kimodo Load Motion — File browser widget for BVH/NPZ files
 * Mirrors the TTS-Audio-Suite pattern: STRING widget + JS file upload button
 *
 * Uploads selected files to ComfyUI's /upload/image endpoint (which accepts
 * non-image files) and sets the file_path widget value so the user can
 * simply click "Browse" to pick local motion files.
 */
import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "kimodo.loadmotion",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "Kimodo_LoadMotion") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      // Upload file to ComfyUI input directory, returns the relative path
      const uploadFile = async (file) => {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("type", "input");
        formData.append("subfolder", "motion");

        const resp = await fetch("/upload/image", { method: "POST", body: formData });
        if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
        const result = await resp.json();
        return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
      };

      // Trigger file picker and update file_path widget
      const browseAndSet = async () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".bvh,.npz";
        input.style.display = "none";

        input.onchange = async (e) => {
          const file = e.target.files[0];
          if (!file) return;

          try {
            const path = await uploadFile(file);
            const fw = this.widgets?.find((w) => w.name === "file_path");
            if (fw) {
              fw.value = path;
              if (fw.callback) fw.callback(path);
            }
          } catch (err) {
            console.error("[Kimodo] File upload failed:", err);
          }
        };

        document.body.appendChild(input);
        input.click();
        document.body.removeChild(input);
      };

      // Create a custom widget containing the browse button + drop zone
      const container = document.createElement("div");
      container.style.cssText = `
        display: flex; gap: 4px; align-items: center;
        padding: 2px 0; width: 100%;
      `;

      const browseBtn = document.createElement("button");
      browseBtn.textContent = "Browse";
      browseBtn.style.cssText = `
        padding: 3px 10px; background: #3a3a3a; color: #ccc;
        border: 1px solid #555; border-radius: 4px; cursor: pointer;
        font-size: 11px; white-space: nowrap;
      `;
      browseBtn.title = "Browse for BVH or NPZ motion files";
      browseBtn.onclick = browseAndSet;

      const dropZone = document.createElement("span");
      dropZone.textContent = "or drop .bvh/.npz here";
      dropZone.style.cssText = `
        color: #666; font-size: 10px; font-style: italic; flex: 1;
        user-select: none;
      `;

      container.appendChild(browseBtn);
      container.appendChild(dropZone);

      // Register as a DOM widget so it renders in the node
      const domWidget = this.addDOMWidget("kimodo_browse", "widget", container, {
        getValue() { return ""; },
        setValue(v) {},
      });
      domWidget.computeSize = function (width) {
        return [width || 400, 28];
      };
      domWidget.element = container;
      this._kimodoBrowseWidget = domWidget;

      // Drag-and-drop on the whole node
      this.dragDropHandler = (e) => {
        e.preventDefault();
        const files = e.dataTransfer?.files;
        if (!files?.length) return;
        const file = files[0];
        const ext = "." + file.name.split(".").pop().toLowerCase();
        if (ext !== ".bvh" && ext !== ".npz") return;

        browseBtn.textContent = "Uploading...";
        uploadFile(file)
          .then((path) => {
            const fw = this.widgets?.find((w) => w.name === "file_path");
            if (fw) {
              fw.value = path;
              if (fw.callback) fw.callback(path);
            }
            browseBtn.textContent = "Browse";
          })
          .catch(() => { browseBtn.textContent = "Browse"; });
      };

      // Hooks for drag events on the node
      this.dragoverHandler = (e) => {
        e.preventDefault();
        container.style.background = "rgba(255,255,255,0.05)";
        container.style.borderRadius = "4px";
      };
      this.dragleaveHandler = () => {
        container.style.background = "";
      };

      return r;
    };

    // Attach drag-drop listeners when the node DOM is ready
    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function (graph) {
      const r = onAdded ? onAdded.apply(this, arguments) : undefined;

      setTimeout(() => {
        const el = this.getNodeel ? this.getNodeel() : this.nodeEl?.el;
        if (el) {
          el.addEventListener("drop", (e) => this.dragDropHandler?.(e));
          el.addEventListener("dragover", (e) => this.dragoverHandler?.(e));
          el.addEventListener("dragleave", () => this.dragleaveHandler?.());
        }
      }, 500);

      return r;
    };

    // Cleanup
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      const r = onRemoved ? onRemoved.apply(this, arguments) : undefined;
      const el = this.getNodeel ? this.getNodeel() : this.nodeEl?.el;
      if (el) {
        el.removeEventListener("drop", this.dragDropHandler);
        el.removeEventListener("dragover", this.dragoverHandler);
        el.removeEventListener("dragleave", this.dragleaveHandler);
      }
      return r;
    };
  },
});
