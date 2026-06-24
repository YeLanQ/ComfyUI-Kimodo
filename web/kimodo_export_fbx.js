import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "kimodo.exportfbx",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "Kimodo_ExportFBX") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      const uploadFile = async (file) => {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("type", "input");
        formData.append("subfolder", "fbx");

        const resp = await fetch("/upload/image", { method: "POST", body: formData });
        if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
        const result = await resp.json();
        return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
      };

      const browseAndSet = async () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".fbx,.FBX";
        input.style.display = "none";

        input.onchange = async (e) => {
          const file = e.target.files[0];
          if (!file) return;

          try {
            const path = await uploadFile(file);
            const fw = this.widgets?.find((w) => w.name === "custom_fbx_path");
            if (fw) {
              fw.value = path;
              if (fw.callback) fw.callback(path);
            }
          } catch (err) {
            console.error("[Kimodo] FBX upload error:", err);
            alert(`FBX upload failed: ${err.message}`);
          }
        };

        input.click();
      };

      const btnW = this.addWidget("button", "browse_fbx", "Browse FBX", () => {
        browseAndSet();
      });
      if (btnW) {
        btnW.serialize = false;
      }

      return r;
    };
  },
});
