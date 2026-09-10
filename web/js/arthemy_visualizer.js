import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Helper function to draw rounded rectangle
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

/**
 * Y coordinate of the bottom of a node's widget stack.
 *
 * Every panel in this file used to hardcode `numWidgets * 24 + offset`, which assumes each
 * widget is exactly 24 px tall. Combo boxes, multiline text areas and DOM widgets are not,
 * so any node that gained a widget had its panel drawn partly behind the widget stack and
 * partly outside the node. LiteGraph records the real Y of each widget in `w.last_y` while
 * it draws, so we use that when it is available and only fall back to measured heights.
 */
function widgetsBottomY(node) {
    const widgets = node.widgets || [];
    const wh = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_WIDGET_HEIGHT) || 20;
    let measured = null;
    let stacked = (typeof node.widgets_start_y === "number") ? node.widgets_start_y : 4;

    for (const w of widgets) {
        if (!w || w.hidden || w.type === "converted-widget") continue;
        let h;
        if (typeof w.computeSize === "function") {
            h = (w.computeSize(node.size[0]) || [0, wh])[1] || wh;
        } else if (w.type === "multiline" || w.type === "customtext") {
            h = 60;
        } else {
            h = wh;
        }
        if (typeof w.last_y === "number") {
            measured = Math.max(measured === null ? 0 : measured, w.last_y + h);
        } else {
            stacked += h + 4;
        }
    }
    return Math.round(measured !== null ? measured : stacked);
}

/**
 * Single source of truth for every colour the suite draws.
 *
 * A colour plays one of TWO roles, and the same hex cannot do both well:
 *
 *   DATA  - a stroke or mark drawn ON the near-black panels. It has to be bright to read
 *           against #0a0f1d, so gold #FFD700 is right here.
 *   CHROME- a node's title bar, which is a BACKGROUND with light title text on top of it.
 *           It has to be dark enough for that text, so gold fails badly: 1.40:1 against
 *           white, i.e. unreadable, which is exactly how the CLIP nodes looked.
 *
 * Keeping one hex for both roles was the mistake. Each family now declares both, with the
 * chrome value chosen to land near the Model purple's 5.8:1 contrast so every title bar in
 * the suite reads with the same ease.
 */
const ARTHEMY_PALETTE = {
    model: "#6351cf",        // Purple - Model (Rotator brand colour, and the Visualizer base)
    modelGlow: "rgba(99, 81, 207, 0.4)",
    clip: "#FFD700",         // Gold - CLIP data marks on the dark panels
    clipGlow: "rgba(255, 215, 0, 0.4)",
    lora: "#ff52d4",         // Pink - external LoRA only, kept out of the purple family on purpose
    fiveDModel: "#1d4ed8",   // Deep blue - Model 5D injection
    fiveDClip: "#ea580c",    // Orange - CLIP 5D injection
    rotation: "#10b981",     // Emerald - rotations
};

// Title-bar colours: each one is its family's DATA colour kept at the same hue and dropped
// into a much darker register, so a header reads as "the dark version of that colour"
// rather than as a different colour. Contrast against white title text, measured:
// 10.16 / 5.86 / 7.90 / 10.39 - all far above the 4.5 floor, unlike the raw data colours
// (gold 1.40, pink 2.83), which are tuned for strokes on a near-black panel instead.
const ARTHEMY_CHROME = {
    model: "#3d2ca0",        // deep indigo: hue 248.6 like #6351cf, just much darker
    clip: "#786305",         // hue 49, within 1.6 deg of the data gold #FFD700
    lora: "#96117a",         // hue 315, the dark register of the LoRA pink #ff52d4
    neutral: "#39404f",      // slate for the nodes that genuinely touch both domains
};

// Darken/lighten a "#rrggbb" colour by a multiplicative factor, used to derive a node's
// body fill from its title-bar accent without hand-picking a second colour for every family.
function shadeHex(hex, factor) {
    const n = parseInt(hex.slice(1), 16);
    const r = Math.round(((n >> 16) & 0xff) * factor);
    const g = Math.round(((n >> 8) & 0xff) * factor);
    const b = Math.round((n & 0xff) * factor);
    return `rgb(${r}, ${g}, ${b})`;
}

// Hue angle of a "#rrggbb" colour, so the panels can derive tints from the palette instead
// of hardcoding numbers that would silently drift if a brand colour ever changed.
function hexHue(hex) {
    const n = parseInt(hex.slice(1), 16);
    const r = ((n >> 16) & 0xff) / 255, g = ((n >> 8) & 0xff) / 255, b = (n & 0xff) / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    if (d === 0) return 0;
    let x;
    if (mx === r) x = ((g - b) / d + 6) % 6;
    else if (mx === g) x = (b - r) / d + 2;
    else x = (r - g) / d + 4;
    return x * 60;
}

/**
 * Folds a measured 0-360 hue into a band centred on the domain's own brand hue.
 *
 * The rotation report carries a real measured hue per block, and drawing it raw looked
 * informative until a Model rotation with tensor_x=-10 / tensor_y=20 landed on hue 50 -
 * gold - and painted a MODEL node's tree in the CLIP colour. The measured value still
 * varies the branches, but it can no longer cross into the other domain's identity:
 * Model stays within 204-294 (blue/purple/magenta), CLIP within 6-96 (red/orange/yellow).
 */
const ARTHEMY_HUE_BAND = 45;
function domainBandHue(brandHex, measuredHue) {
    const brand = hexHue(brandHex);
    const m = ((Number(measuredHue) || 0) % 360 + 360) % 360;
    return (brand + (m / 360 - 0.5) * 2 * ARTHEMY_HUE_BAND + 360) % 360;
}

// A node's family is read straight off the coloured glyph already at the front of its
// display name (NODE_DISPLAY_NAME_MAPPINGS in the Python file: 🟪 Model, 🟨 CLIP, 🩷 LoRA,
// 🟪🟨 both). That means a new node only needs to follow the existing convention to be
// tinted correctly - nothing to add here by hand.
//
// Model used to be 🟦, a blue square that did not match the purple it is drawn with
// everywhere else. It now takes the purple square, and LoRA - which held 🟪 - moves to the
// pink heart, the closest glyph to its own #ff52d4.
function arthemyFamilyColor(displayName) {
    const s = (displayName || "").trim();
    // The combo prefix must be tested FIRST: "🟪🟨" also startsWith "🟪".
    if (s.startsWith("🟪🟨")) return ARTHEMY_CHROME.neutral;
    if (s.startsWith("🟨")) return ARTHEMY_CHROME.clip;
    if (s.startsWith("🟪")) return ARTHEMY_CHROME.model;
    if (s.startsWith("🩷")) return ARTHEMY_CHROME.lora;
    return null;
}

// Hoisted to module scope: both beforeRegisterNodeDef (to install the panel) and the
// setup() websocket listener below (to filter which nodes may receive a live report)
// need this same list.
const ROTATOR_NODE_NAMES = [
    "ArthemyKrea2ModelRotator", "ArthemyKrea2ModelChaosRotator", "ArthemyKrea2LatentSpaceRotator",
    "ArthemyKrea2CLIPRotator", "ArthemyKrea2CLIPChaosRotator", "ArthemyKrea2CLIPSpaceRotator"
];

/**
 * Rewrites a legacy 0-360 rotation value into the signed -180..180 range the angle widgets
 * now use. 355 becomes -5: the same rotation, expressed the way the widget can hold it.
 *
 * This is not cosmetic. ComfyUI's validate_inputs REJECTS a prompt whose value exceeds a
 * widget's declared max ("Value 355.0 bigger than max of 180.0") - it does not clamp - so
 * without this migration every workflow saved before the change would simply refuse to run.
 *
 * The raw `widgets_values` array from the serialized workflow is preferred over the live
 * widget value, because some frontends clamp on assignment and 355 would already have
 * become 180 by the time we could read it back.
 */
function migrateSignedAngles(node, info) {
    const widgets = node?.widgets;
    if (!Array.isArray(widgets)) return;
    const stored = Array.isArray(info?.widgets_values) ? info.widgets_values : null;

    widgets.forEach((w, i) => {
        const opts = w?.options;
        // Only widgets that actually became signed: a negative minimum with a max below a
        // full turn. Nothing else in the suite matches, so no name list to keep in sync.
        if (!opts || typeof opts.min !== "number" || typeof opts.max !== "number") return;
        if (opts.min >= 0 || opts.max >= 360) return;

        const raw = (stored && typeof stored[i] === "number") ? stored[i] : w.value;
        if (typeof raw !== "number" || !Number.isFinite(raw)) return;
        if (raw <= opts.max) return;

        const signed = raw - 360;
        if (signed >= opts.min && signed <= opts.max) w.value = signed;
    });
}

/**
 * Signed ratio between the rotation the widgets ask for NOW and the one the last report was
 * measured with. Drives the "draft" state of the 3D panel: the measured structure stays on
 * screen and bends by this factor, so the user can shape it before running.
 *
 * Signed on purpose. Using a magnitude would leave +5 and -5 looking identical, which is
 * precisely the pair the signed-angle range exists to let people compare.
 */
function previewAngleScale(node, report) {
    const params = report?.params;
    const widgets = node?.widgets;
    if (!params || typeof params !== "object" || !Array.isArray(widgets)) return 1;

    let was = 0, now = 0, seen = 0;
    for (const [name, v] of Object.entries(params)) {
        // seed is an identifier, not a magnitude: summing it would swamp everything else.
        if (typeof v !== "number" || name === "seed") continue;
        const w = widgets.find(x => x?.name === name);
        if (!w || w.type === "converted-widget" || typeof w.value !== "number") continue;
        was += v;
        now += w.value;
        seen++;
    }
    if (!seen) return 1;
    if (Math.abs(was) < 1e-6) return Math.abs(now) < 1e-6 ? 1 : (now > 0 ? 2 : -2);
    const ratio = now / was;
    if (!Number.isFinite(ratio)) return 1;
    return Math.max(-3, Math.min(3, ratio));
}

/**
 * The hue a dual rotation would produce from the node's CURRENT widgets.
 *
 * Mirrors the engine exactly - arthemy_geometry_engine.fast_dual_orthogonal_rotation records
 * `hue = (sx + sy*2 + tx*3 + ty*4) % 360` - so a draft is tinted with the colour the next run
 * will really report, not an invention. Returns null on any node that is not a dual rotator,
 * where the caller keeps the measured hue instead.
 */
function dualPreviewHue(node) {
    const val = (n) => {
        const w = node?.widgets?.find(x => x?.name === n);
        return (w && w.type !== "converted-widget" && typeof w.value === "number") ? w.value : null;
    };
    const sx = val("structural_rot_x"), sy = val("structural_rot_y");
    const tx = val("tensor_rot_x"), ty = val("tensor_rot_y");
    if (sx === null || sy === null || tx === null || ty === null) return null;
    return ((sx + sy * 2 + tx * 3 + ty * 4) % 360 + 360) % 360;
}

/**
 * Does a live report still describe the node as it is configured right now?
 *
 * The report carries `params`: the settings the measurement was ACTUALLY taken with,
 * echoed back by the Python side under their widget names. Comparing those against the
 * current widgets is authoritative - snapshotting the widgets when the message lands is
 * not, because by then the user may already have moved a slider mid-run.
 *
 * The one deliberate exception is `seed` while control_after_generate is set to mutate it:
 * ComfyUI bumps the seed *after* each run, so comparing it would send the panel back to
 * preview the instant every report arrived, and the user changed nothing.
 */
function reportMatchesWidgets(node, report) {
    const params = report?.params;
    if (!params || typeof params !== "object") return true;   // older payload: don't fight it
    const widgets = node?.widgets;
    if (!Array.isArray(widgets)) return true;

    const control = widgets.find(w => w?.name === "control_after_generate");
    const seedAutoBumps = !!control && String(control.value ?? "").toLowerCase() !== "fixed";

    for (const [name, was] of Object.entries(params)) {
        if (name === "seed" && seedAutoBumps) continue;
        const w = widgets.find(x => x?.name === name);
        if (!w) continue;                                     // widget not on this node
        // A widget converted to an input keeps a frozen `value` from before the conversion
        // while the real value arrives over the link. Comparing it would pin the panel to
        // "preview (changed)" forever on any node driven by an upstream primitive.
        if (w.type === "converted-widget") continue;
        const now = w.value;
        if (typeof was === "number" && typeof now === "number") {
            // Floats survive a JSON round trip with tiny drift; a real edit is a step apart.
            if (Math.abs(was - now) > 1e-6) return false;
        } else if (String(was) !== String(now)) {
            return false;
        }
    }
    return true;
}

app.registerExtension({
    name: "Arthemy.Krea2Suite",

    // Fires once per session (not once per node type), so this is where we subscribe to the
    // Rotator's side-channel websocket event. The Python side (send_live_rotation_report)
    // pushes real measured per-block data here WITHOUT touching the (obj, info) tuple the
    // Preset Loader and every rotator's own return value still rely on.
    async setup(app) {
        api.addEventListener("arthemy.rotation_report", (event) => {
            const data = event?.detail;
            if (!data || data.node_id === undefined || data.node_id === null) return;
            const node = app.graph.getNodeById(data.node_id) || app.graph.getNodeById(Number(data.node_id));
            if (!node) return;
            const nodeTypeName = node.type || node.comfyClass;
            if (!ROTATOR_NODE_NAMES.includes(nodeTypeName)) return;
            const isCLIP = nodeTypeName.includes("CLIP");
            if (data.domain && data.domain !== (isCLIP ? "clip" : "model")) return;
            // The report carries its own `params`, so nothing has to be snapshotted here:
            // whether it still matches the node is decided at draw time.
            node.rotationReport = data;
            node.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // =====================================================================
        // 0a. LEGACY ANGLE MIGRATION (every node in the suite)
        // =====================================================================
        // Runs on workflow load, before the prompt can be submitted, so a graph saved with
        // 0-360 angles keeps working instead of being rejected by ComfyUI's range check.
        if (nodeData.name && nodeData.name.startsWith("Arthemy") && !nodeType.prototype._arthemy_angle_migration_installed) {
            nodeType.prototype._arthemy_angle_migration_installed = true;
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, arguments);
                try {
                    migrateSignedAngles(this, info);
                } catch (e) {
                    console.warn("[Arthemy] angle migration skipped:", e);
                }
            };
        }

        // =====================================================================
        // 0. AUTOMATIC NODE COLOUR CODING (every node in the suite)
        // =====================================================================
        // Tints the node's own title bar/body on the canvas, not just the custom panels
        // drawn inside a few of them - so Model vs CLIP is readable at a glance even in a
        // busy graph, with zero manual per-node colour picking.
        if (nodeData.name && nodeData.name.startsWith("Arthemy") && !nodeType.prototype._arthemy_family_color_installed) {
            nodeType.prototype._arthemy_family_color_installed = true;
            const accent = arthemyFamilyColor(nodeData.display_name);
            if (accent) {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    onNodeCreated?.apply(this, arguments);
                    this.color = accent;
                    this.bgcolor = shadeHex(accent, 0.22);
                };
            }
        }

        // =====================================================================
        // 1. INTERACTIVE 3D SKELETON NEURAL TREE (ROTATOR NODES)
        // =====================================================================
        if (ROTATOR_NODE_NAMES.includes(nodeData.name)) {
            if (nodeType.prototype._arthemy_3d_skeleton_installed) return;
            nodeType.prototype._arthemy_3d_skeleton_installed = true;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.size = [410, 680];
                this.camYaw = 0.6;
                this.camPitch = 0.35;
                this.targetYaw = 0.6;
                this.targetPitch = 0.35;
            };

            const computeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function (out) {
                const sz = computeSize ? computeSize.apply(this, arguments) : [380, 660];
                sz[0] = Math.max(sz[0], 410);
                // Reserve the 3D panel ON TOP of the widget stack LiteGraph just measured,
                // so adding a widget can never squeeze the panel out of the node.
                sz[1] = Math.max(sz[1] + 256, 620);
                return sz;
            };

            // 3D Perspective Projection (with 3D camera orbit)
            function project3D(x, y, z, cx, cy, fov, yaw, pitch) {
                const cosY = Math.cos(yaw);
                const sinY = Math.sin(yaw);
                const x1 = x * cosY - z * sinY;
                const z1 = x * sinY + z * cosY;

                const cosP = Math.cos(pitch);
                const sinP = Math.sin(pitch);
                const y2 = y * cosP - z1 * sinP;
                const z2 = y * sinP + z1 * cosP;

                const dist = 32 + z2;
                const scale = fov / Math.max(1.0, dist);
                return {
                    x: cx + x1 * scale,
                    y: cy - y2 * scale,
                    z: z2,
                    scale: scale
                };
            }

            // Draw 3D Wireframe Cylinder with true 3D Pitch (tensorY) and Yaw/Spin (tensorX)
            /**
             * One tensor at the tip of a branch, as a small barrel.
             *
             * `tensorY_rad` tilts the barrel's own axis: 0 leaves it parallel to the trunk,
             * +90 deg points it radially outward from the trunk, -90 deg inward - the tensor
             * equivalent of what structural_rot_y does to the branches themselves.
             *
             * `tensorX_rad` spins the barrel about that axis. A regular polygon rotated about
             * its own centre is indistinguishable from itself, so the spin is carried by a
             * SEAM up one side, a SPOKE across the top cap and a short FIN off the rim, all
             * at the spin angle. Without those the rings alone made tensor_rot_x invisible,
             * which is why it used to be legible only as a hue shift.
             */
            function draw3DCylinderTilted(ctx, cx_pos, cy_pos, cz_pos, cx, cy, fov, yaw, pitch,
                                          phi_branch, tensorX_rad, tensorY_rad, radius, height, color) {
                const segments = 12;
                const halfH = height / 2;

                const cosTy = Math.cos(tensorY_rad);
                const sinTy = Math.sin(tensorY_rad);
                const cosPhi = Math.cos(phi_branch);
                const sinPhi = Math.sin(phi_branch);

                // Axis vector
                const ax = sinTy * cosPhi;
                const ay = cosTy;
                const az = sinTy * sinPhi;

                // Cap centers
                const topCenterX = cx_pos + ax * halfH;
                const topCenterY = cy_pos + ay * halfH;
                const topCenterZ = cz_pos + az * halfH;

                const botCenterX = cx_pos - ax * halfH;
                const botCenterY = cy_pos - ay * halfH;
                const botCenterZ = cz_pos - az * halfH;

                // Basis vectors perpendicular to axis for cap rings
                const uX = -sinPhi;
                const uY = 0;
                const uZ = cosPhi;

                const vX = cosTy * cosPhi;
                const vY = -sinTy;
                const vZ = cosTy * sinPhi;

                const topPoints = [];
                const botPoints = [];

                for (let i = 0; i < segments; i++) {
                    const ang = (i / segments) * Math.PI * 2 + tensorX_rad;
                    const cosA = Math.cos(ang) * radius;
                    const sinA = Math.sin(ang) * radius;

                    const rx = uX * cosA + vX * sinA;
                    const ry = uY * cosA + vY * sinA;
                    const rz = uZ * cosA + vZ * sinA;

                    topPoints.push(project3D(topCenterX + rx, topCenterY + ry, topCenterZ + rz, cx, cy, fov, yaw, pitch));
                    botPoints.push(project3D(botCenterX + rx, botCenterY + ry, botCenterZ + rz, cx, cy, fov, yaw, pitch));
                }

                ctx.strokeStyle = color;
                ctx.lineWidth = 1.2;

                // Top ring
                ctx.beginPath();
                for (let i = 0; i < segments; i++) {
                    const p = topPoints[i];
                    if (i === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                }
                ctx.closePath();
                ctx.stroke();

                // Bottom ring
                ctx.beginPath();
                for (let i = 0; i < segments; i++) {
                    const p = botPoints[i];
                    if (i === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                }
                ctx.closePath();
                ctx.stroke();

                // Side vertical edges (four of them, whatever the segment count)
                const edgeStep = Math.max(1, Math.round(segments / 4));
                for (let i = 0; i < segments; i += edgeStep) {
                    ctx.beginPath();
                    ctx.moveTo(topPoints[i].x, topPoints[i].y);
                    ctx.lineTo(botPoints[i].x, botPoints[i].y);
                    ctx.stroke();
                }

                // Spin marker. Ring vertex 0 sits exactly at tensorX_rad, so the seam / spoke /
                // fin all ride the spin angle: turn tensor_rot_x and this hand sweeps around the
                // barrel while the barrel itself stays put - "girando su se stessi", visibly.
                const cosA0 = Math.cos(tensorX_rad);
                const sinA0 = Math.sin(tensorX_rad);
                const r0x = (uX * cosA0 + vX * sinA0);
                const r0y = (uY * cosA0 + vY * sinA0);
                const r0z = (uZ * cosA0 + vZ * sinA0);

                const capTop = project3D(topCenterX, topCenterY, topCenterZ, cx, cy, fov, yaw, pitch);
                const fin = project3D(topCenterX + r0x * radius * 1.95,
                                      topCenterY + r0y * radius * 1.95,
                                      topCenterZ + r0z * radius * 1.95, cx, cy, fov, yaw, pitch);

                // White, like the branch strokes of the generic preview: the marker has to read
                // as a pointer ON the barrel, and in the tip's own hue it disappears into it at
                // the size this panel actually gets on a node.
                ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                ctx.moveTo(botPoints[0].x, botPoints[0].y);   // seam, up the side
                ctx.lineTo(topPoints[0].x, topPoints[0].y);
                ctx.lineTo(fin.x, fin.y);                     // fin, past the rim
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(capTop.x, capTop.y);               // spoke, across the cap
                ctx.lineTo(topPoints[0].x, topPoints[0].y);
                ctx.stroke();
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.2;
            }

            // Draw Foreground 3D Skeleton Tree Canvas
            const origDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx, canvas) {
                if (origDrawForeground) origDrawForeground.apply(this, arguments);
                if (this.flags?.collapsed) return;

                const nodeW = this.size[0];
                const nodeH = this.size[1];

                const isCLIP = nodeData.name.includes("CLIP");
                const isChaos = nodeData.name.includes("Chaos");
                // The Space Rotators are the Compass Rotator nodes: their widgets are
                // style_direction / rotation_angle, NOT the dual rotator's four axes. Reading
                // the wrong ones left both Compass previews permanently flat at 0°/0°.
                const isCompass = nodeData.name.includes("SpaceRotator");

                // Official Brand Colors - same values as the HUD visualizer and the node's own
                // title bar (ARTHEMY_PALETTE), so this panel can't drift out of sync with them.
                const brandColor = isCLIP ? ARTHEMY_PALETTE.clip : ARTHEMY_PALETTE.model;
                const glowColor = isCLIP ? ARTHEMY_PALETTE.clipGlow : ARTHEMY_PALETTE.modelGlow;

                // Extract widget-derived parameters. These are only ever used for the PREVIEW
                // state (before the node has actually run once) - once a real report arrives
                // from the Python side (arthemy.rotation_report) the tree is redrawn from real
                // measured per-block data instead, so a Chaos node's fake seed-derived numbers
                // never masquerade as real angles once the truth is available.
                // `??`, not `||`: 0 is a legitimate value for every one of these widgets, and
                // `|| default` silently replaced a deliberate 0 with the default - a
                // chaos_strength of 0 used to read out as 35%.
                const widgetNum = (name, fallback) => {
                    const w = this.widgets?.find(x => x?.name === name);
                    const v = parseFloat(w?.value);
                    return Number.isFinite(v) ? v : fallback;
                };

                let structX = 0, structY = 0, tensorX = 0, tensorY = 0, chaosStr = 0;
                let compassAngle = 0, compassDir = 180;
                if (isChaos) {
                    chaosStr = widgetNum("chaos_strength", 0.35);
                    const seedVal = widgetNum("seed", 42);
                    structX = (seedVal * 17) % 360;
                    structY = ((seedVal * 31) % 180) - 90;
                    tensorX = (seedVal * 53) % 360;
                    tensorY = ((seedVal * 71) % 180) - 90;
                } else if (isCompass) {
                    // One tilt plus one direction on the colour wheel: the direction sets the
                    // azimuth the branches fan out along, the angle sets how far they tilt.
                    compassAngle = widgetNum("rotation_angle", 15);
                    compassDir = widgetNum("style_direction", 180);
                    structX = compassDir;
                    structY = compassAngle;
                    tensorX = compassDir;
                    tensorY = compassAngle;
                } else {
                    structX = widgetNum("structural_rot_x", 0);
                    structY = widgetNum("structural_rot_y", 0);
                    tensorX = widgetNum("tensor_rot_x", 0);
                    tensorY = widgetNum("tensor_rot_y", 0);
                }

                // Four states.
                //
                // `per_index` is the ONLY thing that may drive real branches: it holds just
                // the blocks/layers this checkpoint actually has AND that really got patched,
                // unlike selected_indices, which can span a much wider static range ("All
                // Blocks (0-27)" even on a 6-block checkpoint).
                //
                //   GENERIC  - never run: the decorative 12-branch spiral from the widgets.
                //   LIVE     - a report whose settings still match the node: real measurements.
                //   DRAFT    - a report whose settings the user has since changed. The measured
                //              STRUCTURE (which blocks, their relative weight) is still the best
                //              description of this model, so it stays on screen and bends to
                //              follow the new widget values, dimmed and clearly not yet applied.
                //              Throwing it away for the generic spiral - as this used to do -
                //              read as the model vanishing the moment you touched a slider.
                //   EMPTY    - it ran and genuinely patched nothing (all 1-D or quantized).
                const report = this.rotationReport;
                const reportIsCurrent = !report || reportMatchesWidgets(this, report);
                const perIndex = Array.isArray(report?.per_index) ? report.per_index : null;
                const hasStructure = !!(perIndex && perIndex.length > 0);
                const isLive = hasStructure && reportIsCurrent;
                const isDraft = hasStructure && !reportIsCurrent;
                const isConfirmedEmpty = !!report && reportIsCurrent && perIndex !== null && perIndex.length === 0;
                const isGeneric = !hasStructure && !isConfirmedEmpty;

                // How far the current widgets have moved from the ones the measurement was
                // taken with. Signed, so flipping +5 to -5 visibly tips the tree the other
                // way instead of leaving it looking identical.
                const draftScale = isDraft ? previewAngleScale(this, report) : 1;
                // Exact mirror of the engine's own hue formula for a dual rotation
                // (arthemy_geometry_engine.py: (sx + sy*2 + tx*3 + ty*4) % 360), so a draft
                // shows the colour the next run will really produce, not a guess.
                const draftHue = isDraft ? dualPreviewHue(this) : null;

                // Dynamic placement: start directly below the real bottom of the widget stack.
                const panelY = widgetsBottomY(this) + 14;
                const panelH = Math.max(240, nodeH - panelY - 16);
                // Grow the node when the widget stack has pushed the 3D panel past the bottom.
                const neededH = panelY + 240 + 16;
                if (this.size[1] < neededH) {
                    this.size[1] = neededH;
                    this.setDirtyCanvas(true, true);
                }
                const panelW = nodeW - 32;
                const cx = 16 + panelW / 2;
                const cy = panelY + panelH / 2 + 10;
                const fov = 540;

                ctx.save();

                // 1. Glassmorphic Background & Clipping Region
                ctx.fillStyle = "#0a0f1d";
                drawRoundedRect(ctx, 16, panelY, panelW, panelH, 10);
                ctx.fill();
                ctx.strokeStyle = brandColor;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Clip all 3D geometry inside the rounded glassmorphic rectangle
                ctx.save();
                drawRoundedRect(ctx, 16, panelY, panelW, panelH, 10);
                ctx.clip();

                // Damping smooth camera interpolation
                if (this.targetYaw !== undefined) {
                    const diffY = this.targetYaw - this.camYaw;
                    const diffP = this.targetPitch - this.camPitch;
                    if (Math.abs(diffY) > 0.001 || Math.abs(diffP) > 0.001) {
                        this.camYaw += diffY * 0.2;
                        this.camPitch += diffP * 0.2;
                        this.setDirtyCanvas(true, false);
                        if (typeof requestAnimationFrame === "function") {
                            requestAnimationFrame(() => this.setDirtyCanvas(true, false));
                        }
                    } else {
                        this.camYaw = this.targetYaw;
                        this.camPitch = this.targetPitch;
                    }
                }
                const yaw = (Number.isFinite(this.camYaw) ? this.camYaw : 0.6);
                const pitch = (Number.isFinite(this.camPitch) ? this.camPitch : 0.35);

                // Anything not yet applied is dimmed as a whole (trunk + branches) so it reads
                // at a glance as "not real yet". A draft sits between the two: bright enough to
                // work with while you shape it, clearly below a confirmed measurement.
                ctx.save();
                ctx.globalAlpha = isGeneric ? 0.38 : (isDraft ? 0.62 : (isConfirmedEmpty ? 0.5 : 1.0));

                // 2. Central Trunk. Fixed length before anything has been measured; afterwards
                // scaled by mean_relative_delta against a typical reference, so a stronger
                // rotation visibly grows the whole tree - and in a draft that scaling follows
                // the widgets, so the tree grows and shrinks as you drag.
                const REFERENCE_RELATIVE_DELTA = 0.15;
                let trunkHalf = 6.0;
                if (hasStructure && report.mean_relative_delta > 0) {
                    const rel = report.mean_relative_delta;
                    trunkHalf = 6.0 * Math.max(0.6, Math.min(1.8, rel / REFERENCE_RELATIVE_DELTA));
                }
                const trunkTop = project3D(0, trunkHalf, 0, cx, cy, fov, yaw, pitch);
                const trunkBot = project3D(0, -trunkHalf, 0, cx, cy, fov, yaw, pitch);

                ctx.strokeStyle = brandColor;
                ctx.lineWidth = 2.8;
                ctx.shadowColor = glowColor;
                ctx.shadowBlur = 10;
                ctx.beginPath();
                ctx.moveTo(trunkBot.x, trunkBot.y);
                ctx.lineTo(trunkTop.x, trunkTop.y);
                ctx.stroke();
                ctx.shadowBlur = 0;

                if (isGeneric) {
                    // 3a. GENERIC: the fixed 12-branch spiral driven by widget values, used only
                    // until the node has run once - it never claims to be measured data.
                    const numBranches = 12;
                    const giriSpirale = 2;
                    const branchLen = 4.2;
                    const radSX = (structX * Math.PI) / 180.0;
                    const radSY = (structY * Math.PI) / 180.0;
                    const radTX = (tensorX * Math.PI) / 180.0;
                    const radTY = (tensorY * Math.PI) / 180.0;

                    const cosSy = Math.cos(radSY);
                    const sinSy = Math.sin(radSY);

                    for (let i = 0; i < numBranches; i++) {
                        const yStart = -5.0 + (10.0 / (numBranches + 1)) * (i + 1);
                        const spiralAngle = (i / numBranches) * Math.PI * 2 * giriSpirale;
                        const totalAzimuth = spiralAngle + radSX;

                        const xEnd = Math.cos(totalAzimuth) * branchLen * cosSy;
                        const zEnd = Math.sin(totalAzimuth) * branchLen * cosSy;
                        const yEnd = yStart + branchLen * sinSy;

                        const pStart = project3D(0, yStart, 0, cx, cy, fov, yaw, pitch);
                        const pEnd = project3D(xEnd, yEnd, zEnd, cx, cy, fov, yaw, pitch);

                        ctx.strokeStyle = "rgba(255, 255, 255, 0.80)";
                        ctx.lineWidth = 1.4;
                        ctx.beginPath();
                        ctx.moveTo(pStart.x, pStart.y);
                        ctx.lineTo(pEnd.x, pEnd.y);
                        ctx.stroke();

                        draw3DCylinderTilted(
                            ctx, xEnd, yEnd, zEnd, cx, cy, fov, yaw, pitch,
                            totalAzimuth, radTX + spiralAngle, radTY, 0.55, 1.35, brandColor
                        );
                    }
                } else if (hasStructure) {
                    // 3b. MEASURED STRUCTURE: one branch per genuinely-patched block/layer
                    // (per_index), never per selected_indices - a checkpoint smaller than the
                    // stock target range must not grow phantom branches for blocks it lacks.
                    //
                    // The same code draws the LIVE and the DRAFT state. A draft keeps the real
                    // structure and bends it by draftScale, so the user shapes the actual model
                    // instead of watching it be replaced by a generic decoration.
                    const numBranches = perIndex.length;
                    const giriSpirale = 2;
                    const meanRelDelta = report.mean_relative_delta || 0;

                    // The four axes the measurement was taken with, derived from the report by
                    // exactly the rules the widget block above uses for a draft - a Chaos node
                    // from its seed, a Compass from direction/tilt, a dual rotator from its own
                    // four widgets. Same rule on both sides is what keeps LIVE and DRAFT from
                    // snapping to a different shape the instant a slider moves.
                    const reportParams = report?.params || {};
                    let repSX = 0, repSY = 0, repTX = 0, repTY = 0;
                    if (typeof reportParams.chaos_strength === "number") {
                        const seedVal = reportParams.seed ?? 42;
                        repSX = (seedVal * 17) % 360;
                        repSY = ((seedVal * 31) % 180) - 90;
                        repTX = (seedVal * 53) % 360;
                        repTY = ((seedVal * 71) % 180) - 90;
                    } else if (typeof reportParams.style_direction === "number") {
                        repSX = reportParams.style_direction ?? 180;
                        repSY = reportParams.rotation_angle ?? 0;
                        repTX = repSX;
                        repTY = repSY;
                    } else {
                        repSX = reportParams.structural_rot_x ?? 0;
                        repSY = reportParams.structural_rot_y ?? 0;
                        repTX = reportParams.tensor_rot_x ?? 0;
                        repTY = reportParams.tensor_rot_y ?? 0;
                    }
                    const currentSX = isDraft ? structX : repSX;
                    const currentSY = isDraft ? structY : repSY;
                    // Tensor axes now drive GEOMETRY, not just the hue: X spins each tensor on
                    // its own axis, Y tips it outward / inward from the trunk. They used to be
                    // discarded here (the tip got the branch azimuth and half the structural
                    // tilt), so the only trace of them on screen was the colour.
                    const radTX = ((isDraft ? tensorX : repTX) * Math.PI) / 180.0;
                    const radTY = ((isDraft ? tensorY : repTY) * Math.PI) / 180.0;
                    
                    for (let i = 0; i < numBranches; i++) {
                        const block = perIndex[i];
                        const yStart = -5.0 + (10.0 / (numBranches + 1)) * (i + 1);

                        // Azimuth includes the structural X rotation, ensuring no snaps between Live and Draft
                        const spiralAngle = (i / numBranches) * Math.PI * 2 * giriSpirale;
                        const totalAzimuth = spiralAngle + (currentSX * Math.PI / 180.0);

                        // Length: this block's relative_delta against the report's mean, so a
                        // block whose rotation moved the weights more than average visibly
                        // reaches further than one that barely changed.
                        const relRatio = meanRelDelta > 0 ? (block.relative_delta / meanRelDelta) : 1.0;
                        const branchLen = 4.2 * Math.max(0.45, Math.min(1.8, relRatio));

                        // Tilt: driven strictly by current Y widget to prevent Live/Draft snapping. 
                        // The actual measured angle remains in the text readout.
                        const tiltRad = (currentSY * Math.PI) / 180.0;
                        const cosTilt = Math.cos(tiltRad);
                        const sinTilt = Math.sin(tiltRad);

                        const xEnd = Math.cos(totalAzimuth) * branchLen * cosTilt;
                        const zEnd = Math.sin(totalAzimuth) * branchLen * cosTilt;
                        const yEnd = yStart + branchLen * sinTilt;

                        const pStart = project3D(0, yStart, 0, cx, cy, fov, yaw, pitch);
                        const pEnd = project3D(xEnd, yEnd, zEnd, cx, cy, fov, yaw, pitch);

                        // Banded around this domain's own brand hue: the measured value
                        // still separates one block from the next, but a Model tree can
                        // never come out gold (the CLIP colour) the way it used to. In a
                        // draft the hue comes from the widgets via the engine's own formula,
                        // so the colour previews what the next run will actually report.
                        const hue = domainBandHue(brandColor, draftHue !== null ? draftHue : block.hue);
                        const branchColor = `hsla(${hue}, 90%, 72%, 0.9)`;
                        const tipColor = `hsl(${hue}, 85%, 62%)`;

                        ctx.strokeStyle = branchColor;
                        ctx.lineWidth = 1.4;
                        ctx.beginPath();
                        ctx.moveTo(pStart.x, pStart.y);
                        if (block.is_chaos) {
                            // Same zig-zag language the HUD uses for chaos sections, so a Chaos
                            // Rotator's real chaotic blocks read as chaotic here too.
                            const dx = pEnd.x - pStart.x, dy = pEnd.y - pStart.y;
                            const len = Math.hypot(dx, dy) || 1;
                            const nx = -dy / len, ny = dx / len;
                            const amp = 5;
                            ctx.lineTo(pStart.x + dx * 0.33 + nx * amp, pStart.y + dy * 0.33 + ny * amp);
                            ctx.lineTo(pStart.x + dx * 0.66 - nx * amp, pStart.y + dy * 0.66 - ny * amp);
                            ctx.lineTo(pEnd.x, pEnd.y);
                        } else {
                            ctx.lineTo(pEnd.x, pEnd.y);
                        }
                        ctx.stroke();

                        // Same argument order as the generic preview above, so the two states
                        // draw the same tensor from the same numbers: branch azimuth, spin
                        // (tensor X, offset by the branch's own phase along the spiral), tilt
                        // (tensor Y).
                        draw3DCylinderTilted(
                            ctx, xEnd, yEnd, zEnd, cx, cy, fov, yaw, pitch,
                            totalAzimuth, radTX + spiralAngle, radTY,
                            0.55 * Math.max(0.6, Math.min(1.4, relRatio)), 1.35, tipColor
                        );
                    }
                }
                // isConfirmedEmpty: trunk only (already drawn above, dimmed) - no branches,
                // because none were ever really patched; the footer hint below explains why.

                ctx.restore(); // Restore alpha
                ctx.restore(); // Restore clip region

                // 4. State badge, top-right corner of the panel.
                ctx.font = "9px sans-serif";
                let badgeText, badgeColor;
                if (isDraft) {
                    // The structure on screen is real, the settings are not applied yet: say
                    // exactly that, so nobody mistakes a draft for a measurement or wonders
                    // where the tree went.
                    badgeText = "◐ not applied";
                    badgeColor = "rgba(255, 200, 80, 0.9)";
                } else if (isGeneric) {
                    badgeText = "○ preview";
                    badgeColor = "rgba(255, 255, 255, 0.45)";
                } else if (isConfirmedEmpty) {
                    badgeText = "⚠ nothing patched";
                    badgeColor = "rgba(255, 200, 80, 0.85)";
                } else {
                    badgeText = "● live";
                    badgeColor = ARTHEMY_PALETTE.rotation;
                }
                ctx.fillStyle = badgeColor;
                ctx.fillText(badgeText, 16 + panelW - ctx.measureText(badgeText).width - 8, panelY + 14);

                // 5. Metrics Readout (Drawn above clip region). No panel name/version stamp
                // here on purpose - the node's own coloured title bar already says which
                // panel this is, so the corner is reserved for numbers that actually change.
                ctx.font = "10px monospace";
                ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
                let readout;
                if (isDraft) {
                    // Show what the node is set to NOW, plus how far that is from the run the
                    // structure was measured on - the number that explains the shape change.
                    const base = isChaos
                        ? `Chaos: ${(chaosStr * 100).toFixed(0)}%`
                        : (isCompass
                            ? `Tilt: ${compassAngle.toFixed(1)}° | Dir: ${compassDir.toFixed(0)}°`
                            : `Str: ${structX.toFixed(0)}°/${structY.toFixed(0)}° | Tns: ${tensorX.toFixed(0)}°/${tensorY.toFixed(0)}°`);
                    readout = `${base} | ${draftScale >= 0 ? "" : "inverted "}${Math.abs(draftScale).toFixed(2)}x last run`;
                } else if (isLive) {
                    readout = `Mean: ${report.mean_angle.toFixed(0)}° | Δrel: ${(report.mean_relative_delta * 100).toFixed(1)}% | Patched: ${report.patched}`;
                } else if (isConfirmedEmpty) {
                    readout = `Patched: 0 | Skipped 1D: ${report.skipped_1d || 0} | Skipped quant: ${report.quant_skipped || 0}`;
                } else if (isChaos) {
                    readout = `Chaos: ${(chaosStr * 100).toFixed(0)}% | Str: ${structX.toFixed(0)}°/${structY.toFixed(0)}°`;
                } else if (isCompass) {
                    readout = `Tilt: ${compassAngle.toFixed(1)}° | Dir: ${compassDir.toFixed(0)}°`;
                } else {
                    readout = `Str: ${structX.toFixed(0)}°/${structY.toFixed(0)}° | Tns: ${tensorX.toFixed(0)}°/${tensorY.toFixed(0)}°`;
                }
                ctx.fillText(readout, 26, panelY + 16);

                // 6. Footer Hint
                ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
                ctx.font = "italic 9px sans-serif";
                const hint = isConfirmedEmpty
                    ? "Every candidate tensor was skipped (1D or quantized) - nothing to rotate here"
                    : (isDraft
                        ? "Last run's structure, bent to the current values - run to confirm it"
                        : "Drag to orbit | structural X/Y = branch azimuth / elevation | "
                          + "tensor X/Y = tensor spin / tilt out from the trunk");
                ctx.fillText(hint, (nodeW - ctx.measureText(hint).width) / 2, panelY + panelH - 8);

                ctx.restore();
            };

            // Mouse Drag handling for 3D Camera Orbit
            const handleCameraDrag = function (node, local_pos) {
                const panelY = widgetsBottomY(node) + 14;
                const py = local_pos[1];
                return (py >= panelY && py <= node.size[1] - 14);
            };

            const origMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function (e, local_pos) {
                if (handleCameraDrag(this, local_pos)) {
                    this._dragging_3d_cam = true;
                    this._last_mouse_pos = [local_pos[0], local_pos[1]];
                    return true;
                }
                return origMouseDown ? origMouseDown.apply(this, arguments) : false;
            };

            const origMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function (e, local_pos) {
                if (this._dragging_3d_cam && this._last_mouse_pos) {
                    const dx = local_pos[0] - this._last_mouse_pos[0];
                    const dy = local_pos[1] - this._last_mouse_pos[1];
                    this._last_mouse_pos = [local_pos[0], local_pos[1]];

                    this.targetYaw = (Number.isFinite(this.targetYaw) ? this.targetYaw : 0.6) + dx * 0.015;
                    this.targetPitch = Math.max(-1.4, Math.min(1.4, (Number.isFinite(this.targetPitch) ? this.targetPitch : 0.35) + dy * 0.015));
                    this.setDirtyCanvas(true, true);
                    return true;
                }
                return origMouseMove ? origMouseMove.apply(this, arguments) : false;
            };

            const origMouseUp = nodeType.prototype.onMouseUp;
            nodeType.prototype.onMouseUp = function (e, local_pos) {
                if (this._dragging_3d_cam) {
                    this._dragging_3d_cam = false;
                    return true;
                }
                return origMouseUp ? origMouseUp.apply(this, arguments) : false;
            };
        }

        // =====================================================================
        // 2. REAL-TIME WAVEFORM HUD VISUALIZER (MODEL & CLIP VISUALIZER NODES)
        // =====================================================================
        if (nodeData.name === "ArthemyKrea2ModelVisualizer" || nodeData.name === "ArthemyKrea2CLIPVisualizer") {
            if (nodeType.prototype._arthemy_hud_installed) return;
            nodeType.prototype._arthemy_hud_installed = true;

            // Minimum height of the chart area itself. computeSize reserves this ON TOP of
            // whatever LiteGraph says the widgets need, so the HUD can never be pushed off
            // the bottom of the node or hidden behind the widget stack.
            const CHART_MIN_H = 230;
            const CHART_BOTTOM_PAD = 30;
            const HUD_GAP = 22;

            const hudTopY = (node) => widgetsBottomY(node) + HUD_GAP;

            function requiredHeight(node) {
                return hudTopY(node) + CHART_MIN_H + CHART_BOTTOM_PAD;
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.size = [580, Math.max(440, requiredHeight(this))];
            };

            const computeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function (out) {
                const sz = computeSize ? computeSize.apply(this, arguments) : [300, 80];
                sz[0] = Math.max(sz[0], 520);
                // Reserve the chart area IN ADDITION to the widget stack that LiteGraph just
                // measured, instead of clamping to a fixed 400 px that stopped being enough.
                sz[1] = Math.max(sz[1] + CHART_MIN_H + CHART_BOTTOM_PAD, requiredHeight(this));
                return sz;
            };

            // ComfyUI wraps every UI output value in one list per batch item; ints/floats/
            // strings sent as a single-element list need one level of unwrapping before use.
            const unwrapScalar = (v) => {
                while (Array.isArray(v)) v = v[0];
                return v;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const payload = message?.detail || message;
                if (payload) {
                    let gData = payload.graph_data;
                    while (Array.isArray(gData) && gData.length > 0 && Array.isArray(gData[0])) {
                        gData = gData[0];
                    }
                    if (gData && Array.isArray(gData)) {
                        this.graphData = gData;
                    }
                    const scaleVal = unwrapScalar(payload.scale);
                    this.visualScale = (typeof scaleVal === "number") ? scaleVal : 1.0;

                    const titleVal = unwrapScalar(payload.title);
                    this.hudTitle = titleVal || (nodeData.name.includes("CLIP") ? "Krea-2 CLIP" : "Krea-2 Model");

                    // Real architecture of the loaded checkpoint (Krea2Config.probe), so the
                    // axis can say honestly how much of this fixed-size canvas is real data
                    // for THIS model versus unused padding sized for the stock baseline.
                    const realVal = unwrapScalar(payload.real_count);
                    const baseVal = unwrapScalar(payload.baseline_count);
                    this.realCount = (typeof realVal === "number") ? realVal : null;
                    this.baselineCount = (typeof baseVal === "number") ? baseVal : null;

                    this.setDirtyCanvas(true, true);
                }
            };

            const origDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx, canvas) {
                if (origDrawForeground) origDrawForeground.apply(this, arguments);
                if (this.flags?.collapsed) return;

                const isCLIP = nodeData.name.includes("CLIP");

                // Colour standards - kept identical to render_visualizer_image() in Python so
                // the exported PNG and this widget can never disagree, and to ARTHEMY_PALETTE
                // so this HUD, the Rotator's 3D panel and the node's own title bar all agree too.
                const baseColor = isCLIP ? ARTHEMY_PALETTE.clip : ARTHEMY_PALETTE.model;
                const loraColor = ARTHEMY_PALETTE.lora;
                const fiveDColor = isCLIP ? ARTHEMY_PALETTE.fiveDClip : ARTHEMY_PALETTE.fiveDModel;
                const rotColor = ARTHEMY_PALETTE.rotation;
                const axisColor = "rgba(255, 255, 255, 0.25)";

                const paddingX = 14;
                const topY = hudTopY(this);

                // Grow the node if the widget stack pushed the chart past the bottom edge.
                const needed = topY + CHART_MIN_H + CHART_BOTTOM_PAD;
                if (this.size[1] < needed) {
                    this.size[1] = needed;
                    this.size[0] = Math.max(this.size[0], 520);
                    this.setDirtyCanvas(true, true);
                }

                const nodeW = this.size[0];
                const nodeH = this.size[1];
                const chartW = nodeW - (paddingX * 2);
                const chartH = Math.max(CHART_MIN_H, nodeH - topY - CHART_BOTTOM_PAD);
                const centerY = topY + (chartH / 2);

                ctx.save();

                // 1. Container background
                ctx.fillStyle = "#0f172a";
                drawRoundedRect(ctx, paddingX - 4, topY - 18, chartW + 8, chartH + 30, 8);
                ctx.fill();
                ctx.strokeStyle = baseColor;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // 2. Header title, plus a probed-architecture note when this checkpoint's
                // real block/layer count differs from the stock baseline the canvas is
                // sized for (Krea2Config.probe, via the node's execution result).
                ctx.fillStyle = baseColor;
                ctx.font = "bold 11px sans-serif";
                const titleStr = this.hudTitle || (isCLIP ? "Krea-2 CLIP" : "Krea-2 Model");
                ctx.fillText(titleStr, paddingX, topY - 5);
                const titleWidth = ctx.measureText(titleStr).width;
                const hasRealCount = (typeof this.realCount === "number" && typeof this.baselineCount === "number");
                if (hasRealCount && this.realCount !== this.baselineCount) {
                    const noun = isCLIP ? "layers" : "blocks";
                    const note = ` · ${this.realCount}/${this.baselineCount} ${noun} probed`;
                    ctx.font = "9px sans-serif";
                    ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
                    ctx.fillText(note, paddingX + titleWidth + 2, topY - 5);
                }

                // 3. Header legend, laid out right-to-left from the scale badge
                const legendY = topY - 5;
                ctx.font = "9px sans-serif";
                let badgeX = nodeW - paddingX - 58;

                ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
                ctx.fillText(`Scale: ${this.visualScale || 1.0}x`, badgeX, legendY);

                const legend = [
                    { color: fiveDColor, label: "5D", width: 40 },
                    { color: rotColor, label: "Rotation", width: 62 },
                    { color: loraColor, label: "LoRA", width: 45 },
                    { color: baseColor, label: "Base", width: 45 },
                ];
                for (const item of legend) {
                    badgeX -= item.width;
                    ctx.fillStyle = item.color;
                    ctx.fillRect(badgeX, legendY - 6, 7, 7);
                    ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
                    ctx.fillText(item.label, badgeX + 10, legendY);
                }

                // 4. Baseline axis
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
                    const hint = "Run the workflow to display the live patch waveform...";
                    ctx.fillText(hint, paddingX + (chartW - ctx.measureText(hint).width) / 2, centerY + 4);
                    ctx.restore();
                    return;
                }

                // 5. Waveform + per-section indicators
                const data = this.graphData;
                const stepX = chartW / data.length;
                const maxOffsetSpan = (chartH / 2) - 8;
                const userScale = Math.min(99.0, Math.max(0.1, this.visualScale || 1.0));
                const scaleFactor = (maxOffsetSpan / 0.50) * (userScale * 0.35);

                // Sections from realCount..baselineCount are block/layer slots this fixed-size
                // canvas reserves for the stock architecture but that this checkpoint doesn't
                // actually have (see Krea2Config.probe); the trailing special sections (Text
                // Fusion, Time Embed, Projection, Embedding) live past baselineCount and are
                // unaffected. Band them out so they read as "not present", not as real zero data.
                if (hasRealCount && this.realCount < this.baselineCount) {
                    const maxPadX = paddingX + chartW;
                    const padX1 = Math.max(paddingX, Math.min(maxPadX, paddingX + this.realCount * stepX));
                    const padX2 = Math.max(paddingX, Math.min(maxPadX, paddingX + this.baselineCount * stepX));
                    if (padX2 > padX1) {
                        ctx.fillStyle = "rgba(0, 0, 0, 0.28)";
                        ctx.fillRect(padX1, topY - 14, padX2 - padX1, chartH + 20);
                    }
                }

                // Rotation badges are drawn in their own band at the top of the chart, and 5D
                // tags in a band at the bottom, so neither can overlap the waveform.
                const rotBandY = topY + 12;
                const fiveBandY = topY + chartH - 12;

                data.forEach((sec, idx) => {
                    const x1 = paddingX + (idx * stepX);
                    const x2 = x1 + stepX;
                    const isPadding = hasRealCount && idx >= this.realCount && idx < this.baselineCount;

                    const rawOffset = sec.offset || 0.0;
                    const clampedShift = Math.max(-maxOffsetSpan, Math.min(maxOffsetSpan, rawOffset * scaleFactor));
                    const yVal = centerY - clampedShift;

                    const isLora = !!sec.is_lora;
                    const is5D = !!sec.is_5d;
                    const isChaos = !!sec.is_chaos;
                    const rotAngle = sec.rotation_angle || 0.0;
                    const isRot = !!sec.is_rotation || rotAngle !== 0.0;

                    // Priority: purple is reserved for a real, independent LoRA. A 5D harmonic
                    // injection is a slice of a LoRA, not a LoRA, so it gets the domain colour.
                    // A padding slot (no real block behind it) is always drawn muted, whatever
                    // it would otherwise have been coloured.
                    const currentColor = isPadding ? "rgba(255, 255, 255, 0.18)"
                        : (isLora ? loraColor : (is5D ? fiveDColor : baseColor));
                    const emphasis = !isPadding && (isLora || is5D);

                    let prevShift = 0;
                    if (idx > 0) {
                        prevShift = Math.max(-maxOffsetSpan, Math.min(maxOffsetSpan, (data[idx - 1].offset || 0.0) * scaleFactor));
                    }
                    const prevY = centerY - prevShift;

                    ctx.strokeStyle = currentColor;
                    ctx.lineWidth = emphasis ? 2.6 : 1.8;
                    ctx.beginPath();
                    ctx.moveTo(x1, prevY);
                    if (isChaos) {
                        const midX = x1 + (stepX / 2);
                        const zigAmp = 6 * (emphasis ? 1.2 : 1.0);
                        ctx.lineTo(x1 + stepX * 0.25, yVal - zigAmp);
                        ctx.lineTo(midX, yVal + zigAmp);
                        ctx.lineTo(x1 + stepX * 0.75, yVal - zigAmp);
                        ctx.lineTo(x2, yVal);
                    } else {
                        ctx.lineTo(x1, yVal);
                        ctx.lineTo(x2, yVal);
                    }
                    ctx.stroke();

                    // Rotation indicator: emerald dot + the ACTUAL angle of this section, with a
                    // hue-tinted stem down to the baseline showing the style direction.
                    if (isRot && rotAngle !== 0.0 && !isPadding) {
                        const midSecX = x1 + (stepX / 2);
                        const badgeY = rotBandY + ((idx % 2) ? 13 : 0);
                        // Banded to this domain's own hue range, exactly like
                        // domain_band_hue() in the Python PNG renderer, so a Model readout
                        // can never sprout gold stems (gold means CLIP everywhere else).
                        const hue = domainBandHue(baseColor, sec.rotation_hue);

                        ctx.strokeStyle = `hsla(${hue}, 100%, 62%, 0.5)`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(midSecX, badgeY + 4);
                        ctx.lineTo(midSecX, centerY);
                        ctx.stroke();

                        ctx.fillStyle = rotColor;
                        ctx.beginPath();
                        ctx.arc(midSecX, badgeY, 3, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.fillStyle = rotColor;
                        ctx.font = "8px sans-serif";
                        const angStr = `${rotAngle.toFixed(0)}°`;
                        ctx.fillText(angStr, midSecX - ctx.measureText(angStr).width / 2, badgeY - 5);
                    }

                    // 5D indicator: filled square in the domain colour + active dimension count.
                    if (is5D && !isPadding) {
                        const midSecX = x1 + (stepX / 2);
                        const tagY = fiveBandY - ((idx % 2) ? 12 : 0);
                        ctx.fillStyle = fiveDColor;
                        ctx.fillRect(midSecX - 3, tagY - 3, 6, 6);
                        const dims = sec.five_d_dims || 0;
                        if (dims) {
                            ctx.font = "8px sans-serif";
                            const t = `${dims}D`;
                            ctx.fillText(t, midSecX - ctx.measureText(t).width / 2, tagY - 6);
                        }
                    }
                });

                // 6. X-axis section labels
                const axisY = topY + chartH + 11;
                ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
                ctx.font = "8px sans-serif";

                const groups = !isCLIP ? [
                    { label: "B1", endIdx: 4 }, { label: "B2", endIdx: 9 },
                    { label: "B3", endIdx: 14 }, { label: "B4", endIdx: 19 },
                    { label: "B5", endIdx: 23 }, { label: "B6", endIdx: 27 },
                    { label: "TF/TE/PR", endIdx: 30 }
                ] : [
                    { label: "L1", endIdx: 4 }, { label: "L2", endIdx: 9 },
                    { label: "L3", endIdx: 14 }, { label: "L4", endIdx: 19 },
                    { label: "L5", endIdx: 24 }, { label: "L6", endIdx: 29 },
                    { label: "L7", endIdx: 35 }, { label: "EM", endIdx: 36 }
                ];

                let startIdx = 0;
                groups.forEach(g => {
                    if (startIdx >= data.length) return;
                    const endX = paddingX + Math.min(g.endIdx + 1, data.length) * stepX;
                    const startX = paddingX + startIdx * stepX;
                    const midX = (startX + endX) / 2;
                    ctx.fillText(g.label, midX - ctx.measureText(g.label).width / 2, axisY);

                    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(endX, topY + chartH - 4);
                    ctx.lineTo(endX, topY + chartH + 4);
                    ctx.stroke();

                    startIdx = g.endIdx + 1;
                });

                ctx.restore();
            };
        }
    }
});
