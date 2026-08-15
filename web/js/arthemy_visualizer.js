import { app } from "/scripts/app.js";

// Safe rounded rectangle helper for canvas
function drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

app.registerExtension({
    name: "Arthemy.Krea2Visualizer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ArthemyKrea2ModelVisualizer" || nodeData.name === "ArthemyKrea2CLIPVisualizer") {
            
            // IDEMPOTENCY GUARD: Prevent duplicate wrapping if extension is reloaded or loaded from multiple paths
            if (nodeType.prototype._arthemy_hud_installed) {
                return;
            }
            nodeType.prototype._arthemy_hud_installed = true;

            // Set minimum node size for optimal HUD rendering (480x240)
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.size = [480, 240];
            };

            // Override computeSize to ensure LiteGraph reserves space for HUD canvas
            const computeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function (out) {
                const sz = computeSize ? computeSize.apply(this, arguments) : [300, 80];
                sz[0] = Math.max(sz[0], 480);
                sz[1] = Math.max(sz[1], 240);
                return sz;
            };

            // Intercept backend output payload from Python
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const payload = message?.detail || message;
                if (payload && payload.graph_data) {
                    this.graphData = payload.graph_data;
                    const scaleVal = Array.isArray(payload.scale) ? payload.scale[0] : payload.scale;
                    this.visualScale = scaleVal || 1.0;
                    const titleVal = Array.isArray(payload.title) ? payload.title[0] : payload.title;
                    this.hudTitle = titleVal || (nodeData.name.includes("CLIP") ? "Qwen3 Text Encoder" : "Krea-2 Model");
                    this.setDirtyCanvas(true, true);
                }
            };

            // Custom LiteGraph Canvas Foreground Renderer
            const origDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx, canvas) {
                if (origDrawForeground) {
                    origDrawForeground.apply(this, arguments);
                }

                if (this.flags?.collapsed) return;

                // Ensure node size is expanded if too short
                if (this.size[1] < 230) {
                    this.size[1] = 240;
                    this.size[0] = Math.max(this.size[0], 480);
                    this.setDirtyCanvas(true, true);
                }

                const isCLIP = nodeData.name.includes("CLIP");
                const baseColor = isCLIP ? "#FFD700" : "#00E5FF"; // Gold for CLIP, Cyan for Model
                const loraColor = "#D000FF"; // Neon Purple for LoRA
                const axisColor = "rgba(255, 255, 255, 0.25)";

                const paddingX = 14;
                const topY = 80; // Offset comfortably below LiteGraph widgets (scale, image_width, image_height)
                const nodeW = this.size[0];
                const nodeH = this.size[1];
                const chartW = nodeW - (paddingX * 2);
                const chartH = Math.max(80, nodeH - topY - 38);
                const centerY = topY + (chartH / 2);

                ctx.save();

                // Reset canvas shadow state completely before rendering HUD
                ctx.shadowColor = "transparent";
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;

                // 1. Render Solid Glassmorphic Container Background (Wipes previous frame artifacts)
                ctx.fillStyle = "#0f172a"; // Solid dark slate background
                drawRoundedRect(ctx, paddingX - 4, topY - 18, chartW + 8, chartH + 34, 8);
                ctx.fill();

                ctx.strokeStyle = baseColor;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // 2. HUD Header (Left Title)
                ctx.fillStyle = baseColor;
                ctx.font = "bold 11px sans-serif";
                ctx.fillText(this.hudTitle || (isCLIP ? "Qwen3 Text Encoder" : "Krea-2 Model"), paddingX, topY - 4);

                // 3. HUD Header (Right Legend & Scale Badge)
                const legendY = topY - 4;
                ctx.font = "9px sans-serif";
                let badgeX = nodeW - paddingX - 60;

                // Scale Badge
                ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
                ctx.fillText(`Scale: ${this.visualScale || 1.0}x`, badgeX, legendY);

                // Legend items
                badgeX -= 55;
                ctx.strokeStyle = "#FF3366";
                ctx.beginPath();
                ctx.moveTo(badgeX, legendY - 2);
                ctx.lineTo(badgeX + 4, legendY - 6);
                ctx.lineTo(badgeX + 8, legendY);
                ctx.stroke();
                ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
                ctx.fillText("Chaos", badgeX + 11, legendY);

                badgeX -= 45;
                ctx.fillStyle = loraColor;
                ctx.fillRect(badgeX, legendY - 6, 7, 7);
                ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
                ctx.fillText("LoRA", badgeX + 10, legendY);

                badgeX -= 65;
                ctx.fillStyle = baseColor;
                ctx.fillRect(badgeX, legendY - 6, 7, 7);
                ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
                ctx.fillText("Base", badgeX + 10, legendY);

                // 4. Draw Grid & Zero Baseline Axis
                ctx.strokeStyle = axisColor;
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(paddingX, centerY);
                ctx.lineTo(paddingX + chartW, centerY);
                ctx.stroke();
                ctx.setLineDash([]);

                if (!this.graphData || this.graphData.length === 0) {
                    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
                    ctx.font = "italic 11px sans-serif";
                    const hint = "Run workflow to display real-time patch waveform...";
                    ctx.fillText(hint, paddingX + (chartW - ctx.measureText(hint).width) / 2, centerY + 4);
                    ctx.restore();
                    return;
                }

                // 5. Render Section Graph Waveform
                const data = this.graphData;
                const stepX = chartW / data.length;
                
                const maxOffsetSpan = (chartH / 2) - 6; // Max pixel deflection
                const userScale = Math.min(10.0, Math.max(0.1, this.visualScale || 1.0));
                const scaleFactor = (maxOffsetSpan / 0.50) * (userScale * 0.35);

                data.forEach((sec, idx) => {
                    const x1 = paddingX + (idx * stepX);
                    const x2 = x1 + stepX;

                    const rawOffset = sec.offset || 0.0;
                    const pixelShift = rawOffset * scaleFactor;
                    const clampedShift = Math.max(-maxOffsetSpan, Math.min(maxOffsetSpan, pixelShift));
                    const yVal = centerY - clampedShift;

                    const currentColor = sec.is_lora ? loraColor : baseColor;

                    ctx.strokeStyle = currentColor;
                    ctx.lineWidth = sec.is_lora ? 2.5 : 2.0;

                    // Configure glow shadow strictly for line rendering
                    if (sec.is_lora || Math.abs(rawOffset) > 0.001) {
                        ctx.shadowColor = currentColor;
                        ctx.shadowBlur = sec.is_lora ? 8 : 4;
                    } else {
                        ctx.shadowColor = "transparent";
                        ctx.shadowBlur = 0;
                    }

                    ctx.beginPath();
                    let prevShift = 0;
                    if (idx > 0) {
                        prevShift = Math.max(-maxOffsetSpan, Math.min(maxOffsetSpan, (data[idx - 1].offset || 0.0) * scaleFactor));
                    }
                    const prevY = centerY - prevShift;

                    ctx.moveTo(x1, prevY);

                    if (sec.is_chaos) {
                        const midX = x1 + (stepX / 2);
                        const zigAmp = 6 * (sec.is_lora ? 1.2 : 1.0);
                        ctx.lineTo(x1 + stepX * 0.25, yVal - zigAmp);
                        ctx.lineTo(midX, yVal + zigAmp);
                        ctx.lineTo(x1 + stepX * 0.75, yVal - zigAmp);
                        ctx.lineTo(x2, yVal);
                    } else {
                        ctx.lineTo(x1, yVal);
                        ctx.lineTo(x2, yVal);
                    }
                    ctx.stroke();

                    // Immediately clear shadow after line stroke to prevent bleeding onto text
                    ctx.shadowColor = "transparent";
                    ctx.shadowBlur = 0;
                });

                // Ensure shadows are completely disabled before rendering text labels
                ctx.shadowColor = "transparent";
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;

                // 6. Draw Simplified X-Axis Labels (Prevents Text Overlap & Shadow Leaking)
                ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
                ctx.font = "9px sans-serif";
                const axisY = topY + chartH + 12;

                const groups = !isCLIP ? [
                    { label: "B1", endIdx: 4 }, { label: "B2", endIdx: 9 },
                    { label: "B3", endIdx: 14 }, { label: "B4", endIdx: 19 },
                    { label: "B5", endIdx: 23 }, { label: "B6", endIdx: 27 },
                    { label: "TF / TE / PR", endIdx: 30 }
                ] : [
                    { label: "L1", endIdx: 4 }, { label: "L2", endIdx: 9 },
                    { label: "L3", endIdx: 14 }, { label: "L4", endIdx: 19 },
                    { label: "L5", endIdx: 24 }, { label: "L6", endIdx: 29 },
                    { label: "L7", endIdx: 35 }, { label: "EM", endIdx: 36 }
                ];

                let startIdx = 0;
                groups.forEach(g => {
                    const startX = paddingX + (startIdx * stepX);
                    const endX = paddingX + ((g.endIdx + 1) * stepX);
                    const midX = (startX + endX) / 2;
                    const txtW = ctx.measureText(g.label).width;

                    ctx.fillText(g.label, Math.round(midX - (txtW / 2)), axisY);

                    // Vertical separator tick line
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(Math.round(endX), topY + chartH - 2);
                    ctx.lineTo(Math.round(endX), topY + chartH + 5);
                    ctx.stroke();

                    startIdx = g.endIdx + 1;
                });

                ctx.restore();
            };
        }
    }
});
