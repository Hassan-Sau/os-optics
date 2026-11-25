/* OpenSci local viewer: draws one view document (opensci.local_viewer.scene 1.0) with the bundled three.js.
 * Geometry is drawn exactly as given (world frame, millimetres, +Z up). The optional "Spaced" display mode of an assembly
 * moves whole support chains apart for readability and says so on screen; "Exact geometry" is the default.
 */
(function () {
  "use strict";
  var T = window.OPENSCI_THREE;

  var I18N = {
    en: {
      OPTICAL_SIMULATION_VIEW: "Optical Simulation View", EXPERIMENTAL_LAB_ASSEMBLY: "Experimental Lab Assembly",
      views: "Views", iso: "Iso", path_horizontal: "Path horizontal", top: "Top", front: "Front", side: "Side", beam_axis: "Along beam",
      planes: "Planes", plane_xy: "XY plane", plane_zx: "ZX plane", plane_yz: "YZ plane",
      plane_xy_hint: "Orthographic view of the XY plane: X to the right, Y up", plane_zx_hint: "Orthographic view of the ZX plane: Z to the right, X up",
      plane_yz_hint: "Orthographic view of the YZ plane: Y to the right, Z up", path_horizontal_hint: "Orthographic view square-on to the main beam, which runs left to right",
      rotate: "Rotate", rollMinus: "Roll -15°", rollPlus: "Roll +15°", turnBeam: "Turn about beam",
      turnBeamHint: "While on, dragging across the beam turns the view about it, like rolling a rod under your hand",
      fit: "Fit", fitAll: "Fit all", reset: "Reset", display: "Display", exact: "Exact geometry", spaced: "Spaced",
      labels: "Labels", labels_key: "Labels: key parts", labels_all: "Labels: all", labels_off: "Labels: off", edges: "Outlines", clearAperture: "Clear aperture", clearApertureHint: "Marks the outline of each surface's clear aperture", savePng: "Save PNG", legend: "Legend", beams: "Beams", selection: "Selection",
      notes: "Notes", clickHint: "Click a part to see its id, name and size.",
      PRIMARY: "Main beam", SECONDARY: "Weak branch (leak / ghost)", PUPIL: "Pupil rays", DESIGN_AXIS: "Beam (design axis)",
      REFERENCE: "Beam axis behind a central obstruction (no light)",
      spacedBanner: "Spaced display: support chains are moved apart for readability, so positions are not to scale and beams are hidden. Choose “Exact geometry” to see the real layout.",
      units: "Units: mm", up: "Design up: +Z", frame: "Frame", orbit: "Drag: rotate · right-drag: pan · wheel: zoom",
      id: "Id", name: "Name", sku: "SKU", cls: "Class", size: "Size (x × y × z)", center: "Centre", mm: "mm",
      category: "Category", optical_role: "Optical role", form: "Form", diameter_mm: "Diameter (mm)", size_mm: "Size (mm)", position_mm: "Position (mm)",
      quaternion_wxyz: "Orientation (w, x, y, z)", carried_by: "Carried by", support_relation: "Support", line_ref: "BOM line",
      design_instance_id: "Design component", chain: "Support chain", assembly_order: "Assembly order", plane_kind: "Plane",
      target_ids: "Targets", footprints: "Beam footprints", anchor_kind: "Anchor", wavelength: "Wavelength",
      absorbing_extent: "Absorption", absorb_NONE: "None: a monitor, beams pass through",
      absorb_ACTIVE_AREA: "Detector: absorbs on its active area", absorb_UNBOUNDED_PLANE: "Detector: the whole plane absorbs",
      farPlane: "{id} lies {d} mm from the optics: the default framing shows the optics, “Fit all” shows everything.",
      loadError: "The view could not be loaded.", webgl: "This browser cannot draw 3D (WebGL is unavailable).",
      toward: "toward you", away: "away from you",
      parts: "Parts", surfaces: "Optical surfaces", surface: "Surface", surfaceHint: "Click a surface in the view or its name here; the box shows or hides it.",
      role: "Role", radius_mm: "Radius R (mm)", conic: "Conic k", aspheric_coefficients: "Aspheric A4, A6, ...", linear_sag_coefficient: "Axicon slope (dz/dρ)", cylinder_axis: "Cylinder straight along", sag: "Freeform sag",
      vertex_z_mm: "Vertex z (mm, element frame)", clear_radius_mm: "Clear aperture radius (mm)", clear_half_widths_mm: "Clear aperture half widths (mm)",
      aperture_center_mm: "Aperture centre (mm)", coating_kind: "Coating", medium_before: "Medium before", medium_after: "Medium after",
      solid_medium: "Solid medium", placement: "Placement", origin_mm: "Vertex (mm, element frame)", normal: "Axis",
      chord_tolerance_mm: "Drawn within (mm)", profile_domain_limit_mm: "Profile defined out to (mm)",
      REFRACT: "Refracting", REFLECT: "Reflecting", SPLIT: "Splitting", POLARIZATION_ELEMENT: "Jones plane",
      APERTURE_STOP: "Aperture stop", THIN_LENS: "Thin lens plane", ATTENUATOR: "Attenuator plane", FARADAY_ROTATOR: "Faraday rotator plane", GRATING: "Ruled plane", member_id: "Member", optical_model_id: "Optical model",
      obstruction_radius_mm: "Central obstruction radius (mm)", focal_length_mm: "Focal length (mm)",
      max_deviation_mm: "Largest drawn deviation (mm)", chord_tolerance_met: "Within the drawing tolerance",
      role_source: "light source", role_lens: "lens", role_mirror: "mirror", role_splitter: "beam splitter", role_filter: "filter",
      role_window: "window", role_polarizer: "polarizer", role_waveplate: "waveplate", role_aperture: "aperture or stop", role_prism: "prism",
      role_detector: "detector", role_dump: "beam dump", role_other_optical: "optical element",
      val_AIR: "air", val_COAXIAL: "on the element axis", val_BODY_FACE: "on a face of the body",
      val_BODY_SECTION: "inside the body", val_FREE: "posed in the element frame", val_true: "yes", val_false: "no", val_null: "not known",
      part_SOURCE: "Source", part_OBSERVATION_PLANE: "Observation plane", part_TARGET_PLANE: "Target plane"
    },
    "zh-CN": {
      OPTICAL_SIMULATION_VIEW: "光学仿真视图", EXPERIMENTAL_LAB_ASSEMBLY: "实验室装配视图",
      views: "视角", iso: "轴测", path_horizontal: "光路水平", top: "俯视", front: "前视", side: "侧视", beam_axis: "沿光束",
      planes: "平面", plane_xy: "XY 平面", plane_zx: "ZX 平面", plane_yz: "YZ 平面",
      plane_xy_hint: "XY 平面的正交视图：X 向右，Y 向上", plane_zx_hint: "ZX 平面的正交视图：Z 向右，X 向上",
      plane_yz_hint: "YZ 平面的正交视图：Y 向右，Z 向上", path_horizontal_hint: "正对主光束的正交视图，光束从左向右",
      rotate: "旋转", rollMinus: "滚转 -15°", rollPlus: "滚转 +15°", turnBeam: "绕光束转动",
      turnBeamHint: "开启后，横跨光束拖动即绕光束转动视图，如同用手滚动一根杆",
      fit: "适配", fitAll: "显示全部", reset: "复位", display: "显示", exact: "精确几何", spaced: "间隔显示",
      labels: "标签", labels_key: "标签：关键零件", labels_all: "标签：全部", labels_off: "标签：关闭", edges: "轮廓线", clearAperture: "通光孔径", clearApertureHint: "标出每个表面通光孔径的轮廓", savePng: "保存 PNG", legend: "图例", beams: "光束", selection: "选中对象",
      notes: "说明", clickHint: "点击零件查看编号、名称和尺寸。",
      PRIMARY: "主光束", SECONDARY: "弱分支（泄漏/鬼像）", PUPIL: "光瞳光线", DESIGN_AXIS: "光束（设计光轴）",
      REFERENCE: "中心遮挡后的光束轴（无光）",
      spacedBanner: "间隔显示：为便于辨认，各支撑链被分开摆放，位置不按比例，光束已隐藏。选择“精确几何”查看真实布局。",
      units: "单位：mm", up: "设计坐标向上：+Z", frame: "坐标系", orbit: "拖动：旋转 · 右键拖动：平移 · 滚轮：缩放",
      id: "编号", name: "名称", sku: "SKU", cls: "类别", size: "尺寸（x × y × z）", center: "中心", mm: "mm",
      category: "类别", optical_role: "光学作用", form: "形式", diameter_mm: "直径（mm）", size_mm: "尺寸（mm）", position_mm: "位置（mm）",
      quaternion_wxyz: "姿态（w, x, y, z）", carried_by: "承载于", support_relation: "支撑方式", line_ref: "BOM 行",
      design_instance_id: "设计元件", chain: "支撑链", assembly_order: "装配顺序", plane_kind: "平面",
      target_ids: "目标", footprints: "光斑", anchor_kind: "锚点", wavelength: "波长",
      absorbing_extent: "吸收", absorb_NONE: "不吸收：监视面，光束穿过",
      absorb_ACTIVE_AREA: "探测面：在有效区域内吸收", absorb_UNBOUNDED_PLANE: "探测面：整个平面吸收",
      farPlane: "{id} 距光学元件 {d} mm：默认视野显示光学元件，“显示全部”显示所有内容。",
      loadError: "无法加载视图。", webgl: "此浏览器无法绘制 3D（WebGL 不可用）。",
      toward: "朝向您", away: "背离您",
      parts: "零件", surfaces: "光学表面", surface: "表面", surfaceHint: "在视图中或此处点击表面名称选中表面；方框显示或隐藏该表面。",
      role: "作用", radius_mm: "曲率半径 R（mm）", conic: "圆锥系数 k", aspheric_coefficients: "非球面系数 A4, A6, ...", linear_sag_coefficient: "锥面斜率（dz/dρ）", cylinder_axis: "柱面直线方向", sag: "自由曲面矢高",
      vertex_z_mm: "顶点 z（mm，元件坐标系）", clear_radius_mm: "通光孔径半径（mm）", clear_half_widths_mm: "通光孔径半宽（mm）",
      aperture_center_mm: "孔径中心（mm）", coating_kind: "膜层", medium_before: "前侧介质", medium_after: "后侧介质",
      solid_medium: "实体介质", placement: "位置", origin_mm: "顶点（mm，元件坐标系）", normal: "轴向",
      chord_tolerance_mm: "绘制精度（mm）", profile_domain_limit_mm: "面形定义范围（mm）",
      REFRACT: "折射", REFLECT: "反射", SPLIT: "分束", POLARIZATION_ELEMENT: "琼斯平面",
      APERTURE_STOP: "孔径光阑", THIN_LENS: "薄透镜平面", ATTENUATOR: "衰减片平面", FARADAY_ROTATOR: "法拉第旋转器平面", GRATING: "刻线平面", member_id: "成员", optical_model_id: "光学模型",
      obstruction_radius_mm: "中心遮挡半径（mm）", focal_length_mm: "焦距（mm）",
      max_deviation_mm: "最大绘制偏差（mm）", chord_tolerance_met: "在绘制精度内",
      role_source: "光源", role_lens: "透镜", role_mirror: "反射镜", role_splitter: "分束器", role_filter: "滤光片",
      role_window: "窗口片", role_polarizer: "偏振片", role_waveplate: "波片", role_aperture: "孔径或光阑", role_prism: "棱镜",
      role_detector: "探测器", role_dump: "光束收集器", role_other_optical: "光学元件",
      val_AIR: "空气", val_COAXIAL: "位于元件光轴上", val_BODY_FACE: "位于实体表面", val_BODY_SECTION: "位于实体内部",
      val_FREE: "按元件坐标系放置", val_true: "是", val_false: "否", val_null: "未知",
      part_SOURCE: "光源", part_OBSERVATION_PLANE: "观测面", part_TARGET_PLANE: "目标面"
    }
  };
  var VIEW_ORDER = ["iso", "path_horizontal", "top", "front", "side", "beam_axis"];
  var PLANE_ORDER = ["plane_xy", "plane_zx", "plane_yz"];
  var ROLL_STEP_DEG = 15;
  var TURN_PER_PAGE = 2 * Math.PI;  // a drag across the drawing area's height turns the view about the beam once
  var SPACING_FACTOR = 1.7;
  var NEST_PX = 2.5;
  var BEAM_PX = { PRIMARY: 4, SECONDARY: 2, DESIGN_AXIS: 4 };
  var EDGE_ANGLE_DEG = 30;
  var CLEAR_APERTURE_MARK = 0x3f8cff;
  var FIT_MARGIN = 1.12;

  var lang = "en";
  try {
    var stored = window.localStorage.getItem("opensci-viewer-lang");
    lang = stored || ((navigator.language || "en").toLowerCase().indexOf("zh") === 0 ? "zh-CN" : "en");
  } catch (e) {
    lang = (navigator.language || "en").toLowerCase().indexOf("zh") === 0 ? "zh-CN" : "en";
  }
  function t(key) { return (I18N[lang] && I18N[lang][key]) || I18N.en[key] || key; }
  // A fact's value in words where it is a fixed vocabulary value (medium AIR, placement, yes / no, an optical role);
  // other values (numbers, material ids) are shown as they are.
  function valueText(v) {
    var key = "val_" + String(v);
    if ((I18N[lang] && I18N[lang][key]) || I18N.en[key]) return t(key);
    if (typeof v === "string" && I18N.en["role_" + v]) return t("role_" + v);
    return String(v);
  }
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function srgb(rgb) { return new T.Color().setRGB(rgb[0], rgb[1], rgb[2], T.SRGBColorSpace); }
  function css(rgb) { return "rgb(" + rgb.map(function (c) { return Math.round(c * 255); }).join(",") + ")"; }
  function fmt(v) { return (Math.round(v * 100) / 100).toString(); }
  function b64(s, Kind) {
    var bin = atob(s), bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Kind(bytes.buffer);
  }


  // The material model of viewer/shading.py, for one fragment; the Python module writes the same equations out and supplies every
  // number (the document's "shading"). Directions are in the camera frame (x right, y up, z towards the viewer).
  var STUDIO_VERTEX = [
    "varying vec3 vNormal;",
    "varying vec3 vViewPosition;",
    "void main() {",
    "  vec4 mv = modelViewMatrix * vec4(position, 1.0);",
    "  vViewPosition = -mv.xyz;",
    "  vNormal = normalMatrix * normal;",
    "  gl_Position = projectionMatrix * mv;",
    "}"
  ].join("\n");
  var STUDIO_FRAGMENT = [
    "varying vec3 vNormal;",
    "varying vec3 vViewPosition;",
    "uniform vec3 baseColor, emissive, envFloor, envCeiling, keyLight;",
    "uniform float alpha, metallic, roughness, transmission, ior, thicknessMm, sigma, tint;",
    "uniform float horizonSoftness, behindDim, groundDim, roughnessSpread, keyIntensity, minRoughness, bodyAmbient, bodyDiffuse, exposure;",
    "uniform vec3 lightCentre[LIGHTS];",
    "uniform float lightSoftness[LIGHTS], lightIntensity[LIGHTS];",
    "vec3 environment(vec3 d, float r) {",
    "  vec3 colour = mix(envFloor, envCeiling, smoothstep(-horizonSoftness, horizonSoftness, d.y)) * (behindDim + (1.0 - behindDim) * (0.5 + 0.5 * d.z));",
    "  float total = 0.0;",
    "  for (int i = 0; i < LIGHTS; i++) {",
    "    float soft = lightSoftness[i] + roughnessSpread * r * r;",
    "    total += lightIntensity[i] * lightSoftness[i] / soft * exp((dot(d, lightCentre[i]) - 1.0) / soft);",
    "  }",
    "  return colour + total;",
    "}",
    "float highlight(vec3 n, vec3 v, float r) {",
    "  vec3 h = normalize(keyLight + v);",
    "  float a = max(r, minRoughness); a *= a;",
    "  float a2 = a * a, nh = clamp(dot(n, h), 0.0, 1.0), nl = clamp(dot(n, keyLight), 0.0, 1.0), nv = clamp(dot(n, v), 1e-3, 1.0);",
    "  float k = nh * nh * (a2 - 1.0) + 1.0;",
    "  float vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-3);",
    "  return keyIntensity * a2 / (3.14159265358979 * k * k) * vis * nl;",
    "}",
    "vec3 encode(vec3 linear) {",
    "  vec3 c = 1.0 - exp(-exposure * max(linear, vec3(0.0)));",
    "  return mix(1.055 * pow(max(c, vec3(0.0031308)), vec3(1.0 / 2.4)) - 0.055, c * 12.92, step(c, vec3(0.0031308)));",
    "}",
    "void main() {",
    "  vec3 v = isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(vViewPosition);",
    "  vec3 n = normalize(vNormal);",
    "  if (dot(n, v) < 0.0) n = -n;",
    "  float c = clamp(dot(n, v), 1e-3, 1.0);",
    "  float d = (ior - 1.0) / (ior + 1.0);",
    "  vec3 f0 = mix(vec3(d * d), baseColor, metallic);",
    "  float g = 1.0 - c;",
    "  vec3 f = f0 + (vec3(1.0) - f0) * g * g * g * g * g;",
    "  vec3 mirror = 2.0 * c * n - v;",
    "  vec3 specular = f * (environment(mirror, roughness) + highlight(n, v, roughness)) * (1.0 - groundDim * roughness);",
    "  float light = bodyAmbient + bodyDiffuse * abs(dot(n, keyLight));",
    "  if (transmission > 0.0) {",
    "    float path = thicknessMm / sqrt(max(1.0 - (1.0 - c * c) / (ior * ior), 1e-3));",
    "    float absorbed = tint > 0.0 ? tint : 1.0 - exp(-sigma * path);",
    "    float through = (1.0 - (f.r + f.g + f.b) / 3.0) * (1.0 - absorbed);",
    "    vec3 src = encode(specular + emissive) + absorbed * encode(baseColor * light);",
    "    gl_FragColor = vec4(src, 1.0 - through * (1.0 - min(max(max(src.r, src.g), src.b), 1.0)));",
    "  } else {",
    "    gl_FragColor = vec4(encode(specular + (1.0 - metallic) * baseColor * light + emissive) * alpha, alpha);",
    "  }",
    "}"
  ].join("\n");

  function loadDocument() {
    var inline = document.getElementById("opensci-view-document");
    if (inline) return Promise.resolve(JSON.parse(inline.textContent));
    return fetch("view.json", { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function Viewer(doc) {
    this.doc = doc;
    this.parts = [];
    this.outlines = [];
    this.apertureMeshes = [];
    this.beamObjects = [];
    this.markerObjects = [];
    this.lineMaterials = [];
    this.hiddenClasses = {};
    this.hiddenParts = {};
    this.hiddenSurfaces = {};
    this.autoFrame = null;
    this.hiddenRoles = { PUPIL: false, SECONDARY: false, REFERENCE: false };
    this.labelMode = "key";
    this.showEdges = true;
    this.mode = "exact";
    this.selected = null;
    var fb = doc.focus_bounds, ab = doc.bounds;
    this.focusCenter = new T.Vector3().fromArray(fb.min_mm).add(new T.Vector3().fromArray(fb.max_mm)).multiplyScalar(0.5);
    this.radius = Math.max(new T.Vector3().fromArray(fb.size_mm).length() / 2, 1);
    this.allBox = new T.Box3(new T.Vector3().fromArray(ab.min_mm), new T.Vector3().fromArray(ab.max_mm));
    this.focusBox = new T.Box3(new T.Vector3().fromArray(fb.min_mm), new T.Vector3().fromArray(fb.max_mm));
  }

  Viewer.prototype.init = function () {
    var stage = $("stage");
    var renderer = new T.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = T.SRGBColorSpace;
    renderer.toneMapping = T.NoToneMapping;
    stage.insertBefore(renderer.domElement, stage.firstChild);
    this.renderer = renderer;
    this.scene = new T.Scene();
    this.scene.background = backdropTexture(this.doc.shading.backdrop);
    this.perspective = new T.PerspectiveCamera(this.doc.camera_fov_deg, 1, this.radius * 1e-3, this.radius * 200);
    this.orthographic = new T.OrthographicCamera(-1, 1, 1, -1, this.radius * 1e-3, this.radius * 200);
    this.camera = this.perspective;
    this.camera.up.set(0, 0, 1);
    this.scene.add(this.perspective, this.orthographic);
    var hemi = new T.HemisphereLight(0xffffff, 0x9aa1ab, 2.2);
    hemi.position.set(0, 0, 1);
    this.scene.add(hemi);
    var head = new T.DirectionalLight(0xffffff, 1.6);
    head.position.set(-0.3, 0.5, 1);
    this.camera.add(head);
    head.target.position.set(0, 0, -1);
    this.camera.add(head.target);
    this.head = head;
    this.makeControls();
    this.buildStudio();
    this.buildParts();
    this.buildBeams();
    this.buildMarkers();
    this.bindUi();
    var self = this;
    new ResizeObserver(function () { self.resize(); }).observe(stage);
    this.resize();
    this.setView("iso");
    this.applyHash();
    window.addEventListener("hashchange", function () { self.applyHash(); });
  };

  Viewer.prototype.buildParts = function () {
    var self = this;
    this.doc.parts.forEach(function (part) {
      var group = new T.Group();
      group.name = "part:" + part.id;
      var entry = { doc: part, group: group, meshes: [], edges: [], ghosts: [], labelEl: null, offset: new T.Vector3() };
      part.meshes.forEach(function (m) {
        var geometry = new T.BufferGeometry();
        geometry.setAttribute("position", new T.BufferAttribute(b64(m.positions, Float32Array), 3));
        geometry.setIndex(new T.BufferAttribute(b64(m.indices, Uint32Array), 1));
        if (m.normals) geometry.setAttribute("normal", new T.BufferAttribute(b64(m.normals, Float32Array), 3));
        else geometry.computeVertexNormals();
        var glass = m.alpha < 1 || m.finish.transmission > 0;
        var material = m.finish.studio_lit ? self.studioMaterial(m, glass) : new T.MeshStandardMaterial({
          color: srgb(m.rgb), roughness: 0.75, metalness: 0.0, flatShading: true, side: T.DoubleSide,
          transparent: glass, opacity: m.alpha, depthWrite: !glass,
          polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1
        });
        var mesh = new T.Mesh(geometry, material);
        mesh.userData.partId = part.id;
        mesh.userData.surfaceId = m.surface || null;
        mesh.renderOrder = glass ? 2 : 0;
        group.add(mesh);
        entry.meshes.push(mesh);
        if (m.material === "aperture_outline" && m.finish.studio_lit) {
          // the clear-aperture line: part of the glass, marked on request (the Clear aperture button)
          var mark = new T.MeshBasicMaterial({ color: CLEAR_APERTURE_MARK, side: T.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
          mark.emissive = new T.Color(0x000000);
          mesh.userData.glassMaterial = material;
          mesh.userData.markMaterial = mark;
          self.apertureMeshes.push(mesh);
        }
        if (m.finish.studio_lit && glass && part.outlines) return;  // this surface's outline is the document's (buildOutlines)
        var shading = self.doc.shading;
        var edgeColor = glass ? srgb(shading.glass_edge) : srgb(m.rgb.map(function (c) { return c * 0.38; }));
        var edges = new T.LineSegments(new T.EdgesGeometry(geometry, EDGE_ANGLE_DEG),
          new T.LineBasicMaterial({ color: edgeColor, transparent: glass, opacity: glass ? shading.glass_edge_alpha : 1 }));
        group.add(edges);
        edges.userData.surfaceId = m.surface || null;
        entry.edges.push(edges);
        if (glass && self.doc.view_kind === "EXPERIMENTAL_LAB_ASSEMBLY") {
          // an optic inside its mount: its outline stays visible (dashed) through the mount
          var ghost = new T.LineSegments(edges.geometry, new T.LineDashedMaterial({
            color: 0x2a5d8f, dashSize: self.radius * 0.012, gapSize: self.radius * 0.01, transparent: true, opacity: 0.55,
            depthTest: false, depthWrite: false
          }));
          ghost.computeLineDistances();
          ghost.renderOrder = 5;
          group.add(ghost);
          entry.ghosts.push(ghost);
        }
      });
      (part.outlines || []).forEach(function (o) { self.buildOutline(entry, o); });
      var box = new T.Box3(new T.Vector3().fromArray(part.bounds.min_mm), new T.Vector3().fromArray(part.bounds.max_mm));
      entry.box = box;
      entry.anchor = box.getCenter(new T.Vector3());
      self.scene.add(group);
      self.parts.push(entry);
    });
  };

  // One see-through surface's outline: its real edges always, and per view the edges where it turns away from the eye. That
  // silhouette is the curved outline of a face against what lies behind it (the rasteriser draws the same one).
  Viewer.prototype.buildOutline = function (entry, o) {
    var s = this.doc.shading;
    var points = b64(o.points, Float32Array), normals = b64(o.sides, Float32Array), feature = b64(o.feature, Uint8Array);
    var geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.BufferAttribute(new Float32Array(points.length), 3));
    var line = new T.LineSegments(geometry, new T.LineBasicMaterial({
      color: srgb(s.glass_edge), transparent: true, depthWrite: false, opacity: s.glass_edge_alpha * (o.faint ? s.faint_edge_share : 1)
    }));
    line.frustumCulled = false;
    line.renderOrder = 3;
    line.userData.surfaceId = o.surface || null;
    entry.group.add(line);
    entry.edges.push(line);
    this.outlines.push({ line: line, points: points, normals: normals, feature: feature, group: entry.group });
  };

  Viewer.prototype.updateOutlines = function () {
    var eye = this.camera.position, orthographic = this.camera.isOrthographicCamera;
    var m = this.camera.matrixWorld.elements, bx = m[8], by = m[9], bz = m[10];  // towards the viewer
    this.outlines.forEach(function (o) {
      if (!o.line.visible || !o.group.visible) return;
      var attribute = o.line.geometry.attributes.position, out = attribute.array, count = 0, p = o.points, n = o.normals;
      var ox = o.group.position.x, oy = o.group.position.y, oz = o.group.position.z;
      for (var k = 0; k < o.feature.length; k++) {
        var i = k * 6, keep = o.feature[k] === 1;
        if (!keep) {
          var vx = bx, vy = by, vz = bz;
          if (!orthographic) { vx = eye.x - p[i] - ox; vy = eye.y - p[i + 1] - oy; vz = eye.z - p[i + 2] - oz; }
          var a = n[i] * vx + n[i + 1] * vy + n[i + 2] * vz, b = n[i + 3] * vx + n[i + 4] * vy + n[i + 5] * vz;
          keep = (a < 0) !== (b < 0);
        }
        if (keep) {
          for (var j = 0; j < 6; j++) out[count + j] = p[i + j];
          count += 6;
        }
      }
      o.line.geometry.setDrawRange(0, count / 3);
      attribute.needsUpdate = true;
    });
  };

  Viewer.prototype.setApertureMarked = function (on) {
    this.apertureMarked = on;
    this.apertureMeshes.forEach(function (mesh) { mesh.material = on ? mesh.userData.markMaterial : mesh.userData.glassMaterial; });
    this.requestRender();
  };

  // The uniforms every studio-lit material shares: the numbers of viewer/shading.py.
  Viewer.prototype.buildStudio = function () {
    var s = this.doc.shading;
    this.studio = {
      lightCentre: { value: s.lights.map(function (b) { return new T.Vector3().fromArray(b.centre); }) },
      lightSoftness: { value: s.lights.map(function (b) { return b.softness; }) },
      lightIntensity: { value: s.lights.map(function (b) { return b.intensity; }) },
      roughnessSpread: { value: s.roughness_spread }, horizonSoftness: { value: s.horizon_softness },
      behindDim: { value: s.behind_dim }, groundDim: { value: s.ground_dim },
      envFloor: { value: new T.Vector3().fromArray(s.env_floor) }, envCeiling: { value: new T.Vector3().fromArray(s.env_ceiling) },
      keyLight: { value: new T.Vector3().fromArray(s.key_light) }, keyIntensity: { value: s.key_intensity },
      minRoughness: { value: s.min_roughness }, bodyAmbient: { value: s.body_ambient }, bodyDiffuse: { value: s.body_diffuse },
      exposure: { value: s.exposure }
    };
  };

  // Glass and metal: the mesh's own numbers from the Engine GLB's material, what is blended behind it premultiplied.
  Viewer.prototype.studioMaterial = function (m, seeThrough) {
    var s = this.doc.shading, f = m.finish;
    var linear = srgb(m.rgb);
    var uniforms = Object.assign({
      baseColor: { value: new T.Vector3(linear.r, linear.g, linear.b) }, alpha: { value: m.alpha },
      metallic: { value: f.metallic }, roughness: { value: f.roughness }, transmission: { value: f.transmission },
      ior: { value: f.ior === null ? s.default_ior : f.ior },
      thicknessMm: { value: f.thickness_mm === null ? s.default_thickness_mm : f.thickness_mm },
      sigma: { value: -Math.log(1 - m.alpha * s.glass_absorption) / s.reference_thickness_mm }, tint: { value: f.tint }
    }, this.studio);
    var material = new T.ShaderMaterial({
      defines: { LIGHTS: s.lights.length }, uniforms: uniforms, vertexShader: STUDIO_VERTEX, fragmentShader: STUDIO_FRAGMENT,
      side: T.DoubleSide, transparent: seeThrough, depthWrite: !seeThrough, premultipliedAlpha: true, toneMapped: false,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1
    });
    material.emissive = new T.Color(0x000000);  // what a selection lights up, like the other materials
    uniforms.emissive = { value: material.emissive };
    return material;
  };

  Viewer.prototype.fatLine = function (pts, color, widthPx, opts) {
    var positions = [];
    for (var i = 0; i + 1 < pts.length; i++) {
      positions.push(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
    }
    if (!positions.length) return null;
    var geometry = new T.LineSegmentsGeometry();
    geometry.setPositions(positions);
    var o = opts || {};
    var material = new T.LineMaterial({
      // every beam line is in the transparent queue, so renderOrder alone decides: weak stretches, then main beams, then ghosts
      color: color, linewidth: widthPx, worldUnits: false, transparent: true, opacity: o.opacity || 1,
      dashed: !!o.dashed, dashSize: o.dash || 1, gapSize: o.gap || 1, depthWrite: !o.ghost && !o.under
    });
    if (o.ghost) {
      // drawn only where something opaque is clearly in front of the beam (pulled towards the camera so that the beam's
      // own visible line never counts as "in front")
      material.depthFunc = T.GreaterDepth;
      material.polygonOffset = true;
      material.polygonOffsetFactor = -8;
      material.polygonOffsetUnits = -8;
    }
    material.resolution.set(this.width || 800, this.height || 600);
    this.lineMaterials.push(material);
    var line = new T.LineSegments2(geometry, material);
    if (o.dashed) line.computeLineDistances();
    line.renderOrder = o.ghost ? 4 : (o.under ? 0.5 : 1);
    return line;
  };

  Viewer.prototype.buildBeams = function () {
    var self = this, R = this.radius;
    var colours = [];
    this.doc.beams.forEach(function (beam) {
      var key = beam.rgb.join(",");
      if (colours.indexOf(key) < 0) colours.push(key);
    });
    var nested = Math.min(colours.length, 4);
    // wider colours first, narrower over them: main beams of different colours on one line stay visible as rim and core
    var ordered = this.doc.beams.slice().sort(function (a, b) { return colours.indexOf(a.rgb.join(",")) - colours.indexOf(b.rgb.join(",")); });
    ["SECONDARY", "PRIMARY"].forEach(function (pass) {
      ordered.forEach(function (beam, order) {
        var rank = Math.min(colours.indexOf(beam.rgb.join(",")), nested - 1);
        var color = srgb(beam.rgb);
        var group = new T.Group();
        group.name = "beam:" + beam.id + ":" + pass;
        function add(obj, role) { if (obj) { obj.userData.role = role; obj.userData.beamId = beam.id; group.add(obj); } }
        beam.runs.forEach(function (run) {
          var pts = run.points.map(function (p) { return new T.Vector3().fromArray(p); });
          var role = run.role;
          if (pass === "SECONDARY") {
            if (role === "PUPIL" || role === "SECONDARY_PUPIL") {
              var pupil = new T.Line(new T.BufferGeometry().setFromPoints(pts), new T.LineBasicMaterial({
                color: color, transparent: true, opacity: role === "PUPIL" ? self.doc.beam_styles.PUPIL.alpha : 0.12, depthWrite: false }));
              pupil.renderOrder = 0.4;
              add(pupil, role === "PUPIL" ? "PUPIL" : "SECONDARY");
            } else if (role === "SECONDARY") {
              add(self.fatLine(pts, color, BEAM_PX.SECONDARY, { dashed: true, dash: R * 0.02, gap: R * 0.014, opacity: 0.75, under: true }), "SECONDARY");
            } else if (role === "REFERENCE") {
              // behind a central obstruction the chief ray is the beam's axis, not light: thin, grey, dashed
              add(self.fatLine(pts, new T.Color(0x737884), 1, { dashed: true, dash: R * 0.012, gap: R * 0.012, opacity: 0.7, under: true }), "REFERENCE");
            }
            return;
          }
          if (role !== "PRIMARY" && role !== "DESIGN_AXIS") return;
          var width = BEAM_PX[role] + NEST_PX * (nested - 1 - rank);
          if (beam.casing) {
            var casing = self.doc.shading.beam_casing;
            add(self.fatLine(pts, srgb(casing.rgb), width + casing.extra_px, { opacity: casing.alpha, under: true }), role);
          }
          var line = self.fatLine(pts, color, width, {});
          if (line) line.renderOrder = 1 + order * 0.01;
          add(line, role);
          // where the beam passes inside a part it stays traceable: a dashed faint copy drawn only where it is hidden
          add(self.fatLine(pts, color, Math.max(width * 0.6, 1.5), { ghost: true, dashed: true, dash: R * 0.012, gap: R * 0.01, opacity: 0.45 }), role);
        });
        if (pass === "SECONDARY") {
          (beam.extensions || []).forEach(function (run) {
            var pts = run.points.map(function (p) { return new T.Vector3().fromArray(p); });
            var role = run.role === "PRIMARY" || run.role === "SECONDARY" || run.role === "REFERENCE" ? run.role : "PUPIL";
            add(self.fatLine(pts, role === "REFERENCE" ? new T.Color(0x737884) : color, role === "PUPIL" || role === "REFERENCE" ? 1 : 2,
              { dashed: true, dash: R * 0.02, gap: R * 0.015, opacity: 0.55, under: true }), role);
          });
        }
        self.scene.add(group);
        self.beamObjects.push(group);
      });
    });
  };

  Viewer.prototype.buildMarkers = function () {
    var self = this, R = this.radius;
    var classes = {};
    this.doc.classes.forEach(function (c) { classes[c.key] = c; });
    this.doc.markers.forEach(function (m) {
      var c = classes[m["class"]];
      var geometry = m["class"] === "CUSTOMER_SUPPLIED" ? new T.SphereGeometry(R * 0.025, 20, 14) : new T.CylinderGeometry(R * 0.02, R * 0.02, R * 0.004, 4);
      var mesh = new T.Mesh(geometry, new T.MeshBasicMaterial({ color: srgb(c ? c.rgb : [0.5, 0.5, 0.5]) }));
      mesh.position.fromArray(m.position_mm);
      if (m["class"] !== "CUSTOMER_SUPPLIED") mesh.rotation.x = Math.PI / 2;
      mesh.userData.markerId = m.id;
      self.scene.add(mesh);
      self.markerObjects.push({ doc: m, mesh: mesh, labelEl: null, anchor: mesh.position.clone() });
    });
  };

  // ------------------------------------------------------------------------------------------------ camera

  // OrbitControls takes its rotation basis from camera.up once, when it is created: a view with another up needs new
  // controls, or dragging keeps turning about the old axis (and does nothing at all where that axis is the view direction).
  Viewer.prototype.makeControls = function () {
    var viewer = this, target = this.controls ? this.controls.target.clone() : null;
    if (this.controls) this.controls.dispose();
    this.controls = new T.OrbitControls(this.camera, this.renderer.domElement);
    if (target) this.controls.target.copy(target);
    this.controls.screenSpacePanning = true;
    this.controls.enableRotate = !this.turning;
    this.controls.addEventListener("change", this.requestRender.bind(this));
    this.controls.addEventListener("start", function () { viewer.autoFrame = null; });
    this.controlsUp = this.camera.up.clone();
  };

  Viewer.prototype.frame = function (dir, up, box, aim) {
    // the smallest distance along dir at which all 8 corners of the box are inside the view, plus a margin
    var center = aim ? aim.clone() : box.getCenter(new T.Vector3());
    var d = dir.clone().normalize();
    var right = new T.Vector3().crossVectors(up, d);
    if (right.lengthSq() < 1e-12) right.set(1, 0, 0);
    right.normalize();
    var trueUp = new T.Vector3().crossVectors(d, right).normalize();
    var size = box.getSize(new T.Vector3());
    var distance = Math.max(size.length() * 0.1, 1);
    var orthographic = this.camera.isOrthographicCamera;
    var aspect = this.width && this.height ? this.width / this.height : 1;
    var tanV = orthographic ? 0 : Math.tan(this.camera.fov * Math.PI / 360) / FIT_MARGIN;
    var tanH = tanV * aspect;
    var halfH = 0, halfV = 0, deepest = 0;
    for (var i = 0; i < 8; i++) {
      var c = new T.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(center);
      var along = c.dot(d);
      halfH = Math.max(halfH, Math.abs(c.dot(right)));
      halfV = Math.max(halfV, Math.abs(c.dot(trueUp)));
      deepest = Math.max(deepest, Math.abs(along));
      if (!orthographic) distance = Math.max(distance, Math.abs(c.dot(right)) / tanH + along, Math.abs(c.dot(trueUp)) / tanV + along);
    }
    if (orthographic) {
      // far enough back that nothing is cut off while the view is turned: the box's whole diagonal in front of the camera
      distance = Math.max(distance, size.length() * 2);
      var half = Math.max(halfV, halfH / aspect, 1e-6) * FIT_MARGIN;
      this.camera.top = half;
      this.camera.bottom = -half;
      this.camera.left = -half * aspect;
      this.camera.right = half * aspect;
      this.camera.zoom = 1;
    }
    this.camera.up.copy(up);
    if (!up.equals(this.controlsUp) || this.controls.object !== this.camera) this.makeControls();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(d, distance);
    this.camera.near = distance / 1000;
    this.camera.far = distance * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    // framed by the viewer, not by the user: framed again whenever the drawing area changes size, so a view framed
    // before the page finished its layout (or in a resized window) still shows everything
    this.autoFrame = { dir: d.clone(), up: up.clone(), box: box, aim: aim ? aim.clone() : null };
    this.requestRender();
  };

  Viewer.prototype.setView = function (name) {
    var p = this.doc.camera_presets[name];
    if (!p) return;
    this.currentView = name;
    this.useCamera(p.projection === "orthographic" ? this.orthographic : this.perspective);
    this.frame(new T.Vector3().fromArray(p.view_direction).normalize(), new T.Vector3().fromArray(p.up), this.focusBox,
      name === "beam_axis" ? new T.Vector3().fromArray(p.target_mm) : null);
    document.querySelectorAll("#views button, #planes button").forEach(function (b) { b.classList.toggle("active", b.dataset.view === name); });
  };

  // ------------------------------------------------------------------------------------------------ scripted camera
  //
  // The page's scripting surface is window.__opensciViewer (set once the page has loaded):
  //   setCamera({angle_to_axis_deg, elevation_deg, roll_deg, distance_mm, focus: [ids], projection}) -> {dir, up}
  //   setView(preset)                 one of the document's camera_presets (iso, top, front, ...)
  //   frame(dir, up, box, aim)        THREE.Vector3 view direction (target to eye) and screen up, a THREE.Box3 to fit,
  //                                   an optional THREE.Vector3 to aim at; the camera backs off until the box fits
  //   cameraFromAngles(angle, elevation, roll) -> {dir, up}
  // The address takes the same camera: #camera=ANGLE,ELEVATION[,ROLL[,DISTANCE_MM]]&focus=ID+ID&projection=orthographic.
  // Angles are stated in the document's camera_frame (axis: the main beam's axis in the direction the light travels; side
  // and lift: the level side view and its up), as 'os-optics view --camera' states them for the snapshots: the angle
  // between the line of sight and the axis, the eye turned about the axis from side towards lift, the picture turned
  // clockwise by the roll. The line of sight, the screen-up and the point aimed at depend on these numbers (and the focus) only,
  // not on the window size or the pixel ratio; the eye distance follows the window (the framed box fits it) unless DISTANCE_MM
  // is given, which then fixes the eye position as well.
  Viewer.prototype.cameraFromAngles = function (angle, elevation, roll) {
    var f = this.doc.camera_frame, rad = Math.PI / 180;
    var axis = new T.Vector3().fromArray(f.axis), side = new T.Vector3().fromArray(f.side), lift = new T.Vector3().fromArray(f.lift);
    var a = angle * rad, e = (elevation || 0) * rad, r = (roll || 0) * rad;
    var across = side.clone().multiplyScalar(Math.cos(e)).addScaledVector(lift, Math.sin(e));
    var dir = axis.clone().multiplyScalar(-Math.cos(a)).addScaledVector(across, Math.sin(a)).normalize();
    var level = lift.clone().multiplyScalar(Math.cos(e)).addScaledVector(side, -Math.sin(e));
    var up = level.clone().multiplyScalar(Math.cos(r)).addScaledVector(new T.Vector3().crossVectors(dir, level), Math.sin(r)).normalize();
    return { dir: dir, up: up };
  };

  // The box of the named parts (their published bounds), or null when none of them is in the view.
  Viewer.prototype.partsBox = function (ids) {
    var box = new T.Box3(), found = false;
    this.doc.parts.forEach(function (p) {
      if (ids.indexOf(p.id) < 0) return;
      box.expandByPoint(new T.Vector3().fromArray(p.bounds.min_mm));
      box.expandByPoint(new T.Vector3().fromArray(p.bounds.max_mm));
      found = true;
    });
    return found ? box : null;
  };

  Viewer.prototype.setCamera = function (options) {
    var o = options || {};
    var pose = this.cameraFromAngles(+o.angle_to_axis_deg, +(o.elevation_deg || 0), +(o.roll_deg || 0));
    var box = (o.focus && o.focus.length && this.partsBox(o.focus)) || this.focusBox;
    this.currentView = null;
    document.querySelectorAll("#views button, #planes button").forEach(function (b) { b.classList.remove("active"); });
    this.useCamera(o.projection === "orthographic" ? this.orthographic : this.perspective);
    this.frame(pose.dir, pose.up, box);
    var distance = +o.distance_mm;
    if (distance > 0 && this.camera.isPerspectiveCamera) {  // the eye where asked, on the same line of sight
      this.camera.position.copy(this.controls.target).addScaledVector(pose.dir, distance);
      this.camera.near = distance / 1000;
      this.camera.far = distance * 50;
      this.camera.updateProjectionMatrix();
      this.controls.update();
      this.autoFrame = null;
      this.requestRender();
    }
    return pose;
  };

  // #camera=ANGLE,ELEVATION[,ROLL[,DISTANCE_MM]]&focus=ID+ID&projection=orthographic; true when it set the camera.
  Viewer.prototype.applyHash = function () {
    var params = {};
    (window.location.hash || "").replace(/^#/, "").split("&").forEach(function (pair) {
      var i = pair.indexOf("=");
      if (i > 0) params[decodeURIComponent(pair.slice(0, i))] = decodeURIComponent(pair.slice(i + 1));
    });
    if (!params.camera) return false;
    var v = params.camera.split(",").map(Number);
    if (v.length < 2 || v.length > 4 || v.some(function (x) { return !isFinite(x); })) return false;
    this.setCamera({ angle_to_axis_deg: v[0], elevation_deg: v[1], roll_deg: v[2] || 0, distance_mm: v[3],
      focus: params.focus ? params.focus.split("+") : null, projection: params.projection });
    return true;
  };

  Viewer.prototype.useCamera = function (camera) {
    if (camera === this.camera) return;
    camera.add(this.head);
    camera.add(this.head.target);
    this.camera = camera;
  };

  // Turns the camera about the view direction; dragging then rotates about the new screen up.
  Viewer.prototype.roll = function (degrees) {
    var axis = this.camera.position.clone().sub(this.controls.target).normalize();
    this.camera.up.applyAxisAngle(axis, degrees * Math.PI / 180).normalize();
    this.afterTurn();
  };

  // Turns the camera about the main beam's axis through the view's centre, by the part of a drag across the beam as seen on the
  // screen: the side of the system facing the viewer follows the pointer.
  Viewer.prototype.turnAboutBeam = function (dx, dy) {
    var axis = new T.Vector3().fromArray(this.doc.main_axis);
    var m = this.camera.matrixWorld.elements;
    var sx = axis.x * m[0] + axis.y * m[1] + axis.z * m[2], sy = axis.x * m[4] + axis.y * m[5] + axis.z * m[6];
    var length = Math.hypot(sx, sy);
    if (length < 1e-3) { sx = 1; sy = 0; length = 1; }  // looking along the beam: a horizontal drag turns the view
    var across = (dx * -sy / length + -dy * sx / length) / Math.max(this.height || 1, 1);
    var angle = across * TURN_PER_PAGE;
    var offset = this.camera.position.clone().sub(this.controls.target).applyAxisAngle(axis, angle);
    this.camera.position.copy(this.controls.target).add(offset);
    this.camera.up.applyAxisAngle(axis, angle).normalize();
    this.camera.lookAt(this.controls.target);
    this.camera.updateMatrixWorld();
    this.requestRender();
  };

  // After the camera's up has been changed by hand: new controls for it (see makeControls), and the view is the user's now.
  Viewer.prototype.afterTurn = function () {
    this.autoFrame = null;
    this.currentView = null;
    document.querySelectorAll("#views button, #planes button").forEach(function (b) { b.classList.remove("active"); });
    this.makeControls();
    this.controls.update();
    this.requestRender();
  };

  Viewer.prototype.currentDirection = function () {
    return this.camera.position.clone().sub(this.controls.target).normalize();
  };

  // ------------------------------------------------------------------------------------------------ modes

  Viewer.prototype.setMode = function (mode) {
    this.mode = mode;
    var spaced = mode === "spaced";
    var chains = {};
    this.parts.forEach(function (p) {
      var chain = p.doc.facts && p.doc.facts.chain;
      if (!chain) return;
      var c = chains[chain] || (chains[chain] = new T.Box3());
      c.union(p.box);
    });
    var outline = this.doc.identity && this.doc.identity.breadboard_outline_mm;
    var center = outline ? new T.Vector3((outline[0][0] + outline[1][0]) / 2, (outline[0][1] + outline[1][1]) / 2, 0) : this.focusCenter.clone();
    this.parts.forEach(function (p) {
      var chain = p.doc.facts && p.doc.facts.chain;
      p.offset.set(0, 0, 0);
      if (spaced && chain) {
        var c = chains[chain].getCenter(new T.Vector3());
        p.offset.set((c.x - center.x) * (SPACING_FACTOR - 1), (c.y - center.y) * (SPACING_FACTOR - 1), 0);
      }
      p.group.position.copy(p.offset);
    });
    this.beamObjects.forEach(function (g) { g.visible = !spaced; });
    this.markerObjects.forEach(function (m) { m.mesh.visible = !spaced; });
    $("banner").style.display = spaced ? "block" : "none";
    $("banner").textContent = t("spacedBanner");
    document.querySelectorAll("#modes button").forEach(function (b) { b.classList.toggle("active", b.dataset.mode === mode); });
    this.applyVisibility();
    this.requestRender();
  };

  Viewer.prototype.applyVisibility = function () {
    var self = this;
    this.parts.forEach(function (p) {
      var hidden = !!self.hiddenClasses[p.doc["class"]] || !!self.hiddenParts[p.doc.id];
      var off = self.hiddenSurfaces[p.doc.id] || {};
      p.group.visible = !hidden;
      p.meshes.forEach(function (m) { m.visible = !(m.userData.surfaceId && off[m.userData.surfaceId]); });
      p.edges.forEach(function (e) { e.visible = self.showEdges && !(e.userData.surfaceId && off[e.userData.surfaceId]); });
      p.ghosts.forEach(function (e) { e.visible = self.showEdges; });
    });
    this.markerObjects.forEach(function (m) { m.mesh.visible = self.mode !== "spaced" && !self.hiddenClasses[m.doc["class"]]; });
    this.beamObjects.forEach(function (g) {
      g.children.forEach(function (o) {
        var role = o.userData.role;
        var off = self.hiddenRoles[role] || (role === "PRIMARY" || role === "DESIGN_AXIS" ? self.hiddenRoles.PRIMARY : false);
        o.visible = !off;
      });
    });
    this.requestRender();
  };

  // ------------------------------------------------------------------------------------------------ UI

  Viewer.prototype.bindUi = function () {
    var self = this;
    var views = $("views");
    function viewButton(name, group) {
      var b = el("button", "", "");
      b.dataset.view = name;
      b.dataset.i18n = name;
      if (I18N.en[name + "_hint"]) b.dataset.i18nTitle = name + "_hint";
      b.addEventListener("click", function () { self.setView(name); });
      group.appendChild(b);
    }
    VIEW_ORDER.forEach(function (name) { viewButton(name, views); });
    PLANE_ORDER.forEach(function (name) { viewButton(name, $("planes")); });
    $("roll-minus").addEventListener("click", function () { self.roll(-ROLL_STEP_DEG); });
    $("roll-plus").addEventListener("click", function () { self.roll(ROLL_STEP_DEG); });
    var turn = $("turn-beam");
    if (!this.doc.main_axis) turn.style.display = "none";
    turn.addEventListener("click", function () {
      self.turning = !self.turning;
      turn.classList.toggle("active", self.turning);
      turn.setAttribute("aria-pressed", String(self.turning));
      self.controls.enableRotate = !self.turning;
    });
    var modes = $("modes");
    ["exact"].concat(this.doc.view_kind === "EXPERIMENTAL_LAB_ASSEMBLY" ? ["spaced"] : []).forEach(function (mode) {
      var b = el("button", mode === "exact" ? "active" : "", "");
      b.dataset.mode = mode;
      b.dataset.i18n = mode;
      b.addEventListener("click", function () { self.setMode(mode); });
      modes.appendChild(b);
    });
    $("fit").addEventListener("click", function () { self.frame(self.currentDirection(), self.camera.up.clone(), self.focusBox); });
    $("fit-all").addEventListener("click", function () { self.frame(self.currentDirection(), self.camera.up.clone(), self.allBox); });
    $("reset").addEventListener("click", function () { self.setMode("exact"); self.setView("iso"); });
    $("toggle-labels").addEventListener("click", function (e) {
      self.labelMode = { key: "all", all: "off", off: "key" }[self.labelMode];
      e.currentTarget.classList.toggle("active", self.labelMode !== "off");
      e.currentTarget.textContent = t("labels_" + self.labelMode);
      self.requestRender();
    });
    $("toggle-edges").addEventListener("click", function (e) {
      self.showEdges = !self.showEdges;
      e.currentTarget.classList.toggle("active", self.showEdges);
      self.applyVisibility();
    });
    var aperture = $("toggle-aperture");
    if (!this.apertureMeshes.length) aperture.style.display = "none";
    aperture.addEventListener("click", function () {
      self.setApertureMarked(!self.apertureMarked);
      aperture.classList.toggle("active", self.apertureMarked);
      aperture.setAttribute("aria-pressed", String(!!self.apertureMarked));
    });
    $("snapshot").addEventListener("click", function () {
      self.renderNow();
      var a = el("a");
      a.href = self.renderer.domElement.toDataURL("image/png");
      a.download = "opensci-" + (self.doc.view_kind === "EXPERIMENTAL_LAB_ASSEMBLY" ? "assembly" : "optical") + "-" + (self.currentView || "view") + ".png";
      a.click();
    });
    $("lang").addEventListener("click", function () {
      lang = lang === "en" ? "zh-CN" : "en";
      try { window.localStorage.setItem("opensci-viewer-lang", lang); } catch (e) { /* per-viewer convenience only */ }
      self.translate();
    });
    var canvas = this.renderer.domElement;
    canvas.addEventListener("pointermove", function (e) {
      if (self.turning && self.turnFrom) {
        self.turnAboutBeam(e.clientX - self.turnFrom[0], e.clientY - self.turnFrom[1]);
        self.turnFrom = [e.clientX, e.clientY];
        self.turned = true;
        return;
      }
      self.hover(e);
    });
    canvas.addEventListener("pointerleave", function () { $("tooltip").style.display = "none"; });
    canvas.addEventListener("pointerdown", function (e) {
      self.down = [e.clientX, e.clientY];
      if (self.turning && e.button === 0) self.turnFrom = [e.clientX, e.clientY];
    });
    canvas.addEventListener("pointerup", function (e) {
      if (self.down && Math.hypot(e.clientX - self.down[0], e.clientY - self.down[1]) < 5) self.select(self.pressedLabel || self.pick(e));
      self.down = null;
      self.pressedLabel = null;
      self.turnFrom = null;
      if (self.turned) {
        self.turned = false;
        self.afterTurn();
      }
    });
    this.buildLabels();
    this.translate();
  };

  Viewer.prototype.translate = function () {
    var self = this;
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach(function (n) { n.textContent = t(n.dataset.i18n); });
    document.querySelectorAll("[data-i18n-title]").forEach(function (n) { n.title = t(n.dataset.i18nTitle); });
    $("toggle-labels").textContent = t("labels_" + this.labelMode);
    $("lang").textContent = lang === "en" ? "中文" : "English";
    $("title").textContent = t(this.doc.view_kind);
    document.title = t(this.doc.view_kind) + " · OpenSci";
    $("banner").textContent = t("spacedBanner");
    this.buildLegend();
    var notes = $("notes");
    notes.textContent = "";
    (this.doc.notes || []).forEach(function (n) {
      var text = n.code === "FAR_PLANE" ? t("farPlane").replace("{id}", n.id).replace("{d}", fmt(n.distance_mm)) : (n.text || "");
      notes.appendChild(el("div", "", text));
    });
    var footer = $("footer");
    footer.textContent = "";
    [t("units"), t("up"), t("frame") + ": " + this.doc.frame, t("orbit")].forEach(function (s) { footer.appendChild(el("span", "", s)); });
    Object.keys(this.doc.identity || {}).forEach(function (k) {
      var v = self.doc.identity[k];
      if (typeof v === "string" && /hash/.test(k)) footer.appendChild(el("span", "", k + ": " + v.slice(0, 12)));
    });
    if (this.selected) this.select(this.selected);
  };

  Viewer.prototype.buildLegend = function () {
    var self = this, legend = $("legend"), counts = {};
    legend.textContent = "";
    this.doc.parts.forEach(function (p) { counts[p["class"]] = (counts[p["class"]] || 0) + 1; });
    this.doc.markers.forEach(function (m) { counts[m["class"]] = (counts[m["class"]] || 0) + 1; });
    this.doc.classes.forEach(function (c) {
      var row = el("div", "legend-row"), label = el("label"), box = el("input");
      box.type = "checkbox";
      box.checked = !self.hiddenClasses[c.key];
      box.addEventListener("change", function () { self.hiddenClasses[c.key] = !box.checked; self.applyVisibility(); });
      label.appendChild(box);
      // a class is shown in the colours it is drawn in: a compound unit by its members' (lens glass, stop plate, ...)
      (c.swatches || [c.rgb]).forEach(function (rgb) {
        var sw = el("span", "swatch");
        sw.style.background = css(rgb);
        sw.style.opacity = c.alpha < 1 ? 0.75 : 1;
        label.appendChild(sw);
      });
      label.appendChild(el("span", "", lang === "zh-CN" ? c.zh : c.en));
      row.append(label, el("span", "count", String(counts[c.key] || 0)));
      legend.appendChild(row);
    });
    var parts = $("part-legend");
    if (parts) {
      parts.textContent = "";
      this.doc.parts.forEach(function (p) {
        var row = el("div", "legend-row"), label = el("label"), box = el("input");
        box.type = "checkbox";
        box.checked = !self.hiddenParts[p.id];
        box.addEventListener("change", function () { self.hiddenParts[p.id] = !box.checked; self.applyVisibility(); });
        // sources and planes are named by their class in the page's language; an optic by its own name
        var named = I18N.en["part_" + p["class"]] ? t("part_" + p["class"]) : p.label;
        label.append(box, el("span", "", p.id + (named && named !== p.id ? " · " + named : "")));
        row.appendChild(label);
        parts.appendChild(row);
      });
    }
    var beams = $("beam-legend");
    beams.textContent = "";
    var roles = [];
    var colours = [];
    this.doc.beams.forEach(function (b) {
      b.runs.forEach(function (r) {
        var role = r.role === "SECONDARY_PUPIL" ? "PUPIL" : r.role;
        if (roles.indexOf(role) < 0) roles.push(role);
      });
      if (!colours.some(function (c) { return c.rgb.join() === b.rgb.join(); })) colours.push(b);
    });
    ["PRIMARY", "DESIGN_AXIS", "SECONDARY", "PUPIL", "REFERENCE"].filter(function (r) { return roles.indexOf(r) >= 0; }).forEach(function (role) {
      var row = el("div", "legend-row"), label = el("label"), box = el("input");
      box.type = "checkbox";
      box.checked = !self.hiddenRoles[role === "DESIGN_AXIS" ? "PRIMARY" : role];
      box.addEventListener("change", function () {
        self.hiddenRoles[role === "DESIGN_AXIS" ? "PRIMARY" : role] = !box.checked;
        self.applyVisibility();
      });
      label.appendChild(box);
      var styles = self.doc.beam_styles || {};
      if (role === "REFERENCE") {
        // the reference axis is drawn grey and dashed whatever the beam's colour: one swatch in that style
        var ref = el("span", "swatch line dashed");
        ref.style.borderTopColor = css((styles.REFERENCE && styles.REFERENCE.rgb) || [0.45, 0.47, 0.52]);
        label.appendChild(ref);
      } else {
        colours.forEach(function (b) {
          var sw = el("span", "swatch line" + (role === "SECONDARY" ? " dashed" : ""));
          sw.style.borderTopColor = css(b.rgb);
          sw.title = b.wavelength_nm ? b.wavelength_nm + " nm" : "";
          label.appendChild(sw);
        });
      }
      label.appendChild(el("span", "", t(role)));
      row.appendChild(label);
      beams.appendChild(row);
    });
    colours.forEach(function (b) {
      if (!b.wavelength_nm) return;
      var row = el("div", "legend-row count");
      var sw = el("span", "swatch line");
      sw.style.borderTopColor = css(b.rgb);
      row.append(sw, el("span", "", t("wavelength") + " " + b.wavelength_nm + " nm"));
      beams.appendChild(row);
    });
  };

  Viewer.prototype.buildLabels = function () {
    var self = this, layer = $("labels");
    // A label sits on its part, often in the middle of the drawing: a press on it is a press on the drawing, so a drag from it
    // rotates like any other; without a drag it selects the label's part (the canvas's pointerup).
    function make(id, hit) {
      var d = el("div", "label", id);
      d.addEventListener("pointerdown", function (e) {
        self.pressedLabel = hit;
        self.renderer.domElement.dispatchEvent(new PointerEvent("pointerdown", e));
      });
      layer.appendChild(d);
      return d;
    }
    this.parts.forEach(function (p) {
      if (p.doc["class"] === "BREADBOARD") return;
      p.labelEl = make(p.doc.id, { kind: "part", entry: p });
    });
    this.markerObjects.forEach(function (m) {
      m.labelEl = make(m.doc.id, { kind: "marker", entry: m });
    });
  };

  Viewer.prototype.updateLabels = function () {
    var w = this.renderer.domElement.clientWidth, h = this.renderer.domElement.clientHeight, self = this;
    var placed = [];
    var items = [];
    var key = { OPTIC: 1, CUSTOMER_SUPPLIED: 1, OBSERVATION: 1 };
    var assembly = this.doc.view_kind === "EXPERIMENTAL_LAB_ASSEMBLY";
    var sel = this.selected && this.selected.entry;
    function wanted(entry, cls) {
      if (entry === sel) return true;
      if (self.labelMode === "off") return false;
      return self.labelMode === "all" || !assembly || !!key[cls];
    }
    this.parts.forEach(function (p) {
      if (p.labelEl) items.push({ el: p.labelEl, anchor: p.anchor.clone().add(p.offset), visible: p.group.visible && wanted(p, p.doc["class"]) });
    });
    this.markerObjects.forEach(function (m) {
      if (m.labelEl) items.push({ el: m.labelEl, anchor: m.anchor, visible: m.mesh.visible && wanted(m, m.doc["class"]) });
    });
    items.forEach(function (it) {
      var v = it.anchor.clone().project(self.camera);
      it.x = (v.x + 1) / 2 * w;
      it.y = (1 - v.y) / 2 * h;
      it.show = it.visible && v.z < 1 && v.z > -1;
    });
    items.sort(function (a, b) { return a.y - b.y; });
    items.forEach(function (it) {
      it.el.style.display = it.show ? "block" : "none";
      if (!it.show) return;
      var rw = it.el.offsetWidth || 60, rh = it.el.offsetHeight || 16, y = it.y - 4;
      for (var k = 0; k < 8; k++) {
        var clash = placed.some(function (r) { return Math.abs(r.x - it.x) < (r.w + rw) / 2 && Math.abs(r.y - y) < rh; });
        if (!clash) break;
        y -= rh + 1;
      }
      placed.push({ x: it.x, y: y, w: rw });
      it.el.style.left = it.x + "px";
      it.el.style.top = y + "px";
    });
  };

  Viewer.prototype.updateTriad = function () {
    var svg = $("triad"), m = this.camera.matrixWorld.elements;
    var right = new T.Vector3(m[0], m[1], m[2]), up = new T.Vector3(m[4], m[5], m[6]), back = new T.Vector3(m[8], m[9], m[10]);
    var dark = this.doc.shading.backdrop.dark;
    var axes = [["X", [1, 0, 0], dark ? "#ec8a8a" : "#d92626"], ["Y", [0, 1, 0], dark ? "#83cf92" : "#1a9933"], ["Z", [0, 0, 1], dark ? "#92aef2" : "#2659e6"]].map(function (a) {
      var v = new T.Vector3().fromArray(a[1]);
      return { name: a[0], color: a[2], x: v.dot(right) * 30, y: -v.dot(up) * 30, depth: v.dot(back) };
    });
    axes.sort(function (a, b) { return a.depth - b.depth; });
    var parts = ['<circle r="3" fill="' + (dark ? "#d8dde4" : "#333") + '"/>'];
    axes.forEach(function (a) {
      var short = Math.hypot(a.x, a.y) < 5;
      if (short) {
        parts.push('<circle r="6" fill="none" stroke="' + a.color + '" stroke-width="2"/>' +
          (a.depth > 0 ? '<circle r="2" fill="' + a.color + '"/>' : ''));
        parts.push('<text x="8" y="-8" font-size="11" fill="' + a.color + '">' + a.name + " " + (a.depth > 0 ? t("toward") : t("away")) + "</text>");
        return;
      }
      parts.push('<line x1="0" y1="0" x2="' + a.x + '" y2="' + a.y + '" stroke="' + a.color + '" stroke-width="2.5" stroke-linecap="round"/>');
      parts.push('<text x="' + a.x * 1.3 + '" y="' + (a.y * 1.3 + 4) + '" font-size="12" font-weight="600" text-anchor="middle" fill="' + a.color + '">' + a.name + "</text>");
    });
    svg.innerHTML = parts.join("");
  };

  Viewer.prototype.pick = function (e) {
    var rect = this.renderer.domElement.getBoundingClientRect();
    var ndc = new T.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    var ray = new T.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    var targets = [];
    this.parts.forEach(function (p) {
      if (p.group.visible) targets.push.apply(targets, p.meshes.filter(function (m) { return m.visible; }));
    });
    this.markerObjects.forEach(function (m) { if (m.mesh.visible) targets.push(m.mesh); });
    var hits = ray.intersectObjects(targets, false);
    if (!hits.length) return null;
    var o = hits[0].object;
    if (o.userData.markerId) {
      var m = this.markerObjects.filter(function (x) { return x.mesh === o; })[0];
      return { kind: "marker", entry: m };
    }
    var p = this.parts.filter(function (x) { return x.doc.id === o.userData.partId; })[0];
    return p ? { kind: "part", entry: p, surface: o.userData.surfaceId || null } : null;
  };

  Viewer.prototype.classLabel = function (key) {
    var c = this.doc.classes.filter(function (x) { return x.key === key; })[0];
    return c ? (lang === "zh-CN" ? c.zh : c.en) : key;
  };

  Viewer.prototype.hover = function (e) {
    var self = this;
    if (this.hoverPending) return;
    this.hoverPending = true;
    requestAnimationFrame(function () {
      self.hoverPending = false;
      var hit = self.pick(e), tip = $("tooltip");
      if (!hit) { tip.style.display = "none"; return; }
      var d = hit.entry.doc, lines = [d.id + (d.label && d.label !== d.id ? " · " + d.label : "")];
      if (d.sku) lines.push(d.sku);
      if (hit.kind === "part") {
        var s = d.bounds.size_mm;
        lines.push(self.classLabel(d["class"]) + " · " + fmt(s[0]) + " × " + fmt(s[1]) + " × " + fmt(s[2]) + " mm");
        var sf = self.surfaceFacts(d, hit.surface);
        if (sf) lines.push(self.surfaceLine(sf));
      } else {
        lines.push(self.classLabel(d["class"]));
      }
      tip.textContent = lines.join("\n");
      var rect = $("stage").getBoundingClientRect();
      tip.style.left = (e.clientX - rect.left + 14) + "px";
      tip.style.top = (e.clientY - rect.top + 14) + "px";
      tip.style.display = "block";
    });
  };

  Viewer.prototype.select = function (hit) {
    var self = this;
    this.parts.forEach(function (p) {
      var on = hit && hit.kind === "part" && hit.entry === p;
      p.meshes.forEach(function (m) {
        var picked = on && hit.surface && m.userData.surfaceId === hit.surface;
        m.material.emissive.setHex(picked ? 0x2f6fff : (on ? (hit.surface ? 0x0d2350 : 0x1f4fb8) : 0x000000));
      });
      if (p.labelEl) p.labelEl.classList.toggle("selected", !!on);
    });
    this.markerObjects.forEach(function (m) { if (m.labelEl) m.labelEl.classList.toggle("selected", !!(hit && hit.entry === m)); });
    this.selected = hit;
    var details = $("details");
    details.textContent = "";
    if (!hit) {
      details.appendChild(el("span", "count", t("clickHint")));
      this.requestRender();
      return;
    }
    var d = hit.entry.doc, table = el("table");
    function row(k, v) {
      if (v === undefined || v === null || v === "" || (Array.isArray(v) && !v.length)) return;
      var tr = el("tr");
      var text = Array.isArray(v) ? v.map(function (x) {
        return typeof x === "object" ? Object.keys(x).map(function (kk) { return kk + " " + x[kk]; }).join(", ") : (typeof x === "number" ? fmt(x) : x);
      }).join(Array.isArray(v) && typeof v[0] === "object" ? "; " : ", ") : String(v);
      tr.append(el("td", "", t(k)), el("td", "", text));
      table.appendChild(tr);
    }
    row("id", d.id);
    row("name", d.label);
    row("sku", d.sku);
    row("cls", this.classLabel(d["class"]));
    if (hit.kind === "part") {
      var s = d.bounds.size_mm, mn = d.bounds.min_mm, mx = d.bounds.max_mm;
      row("size", fmt(s[0]) + " × " + fmt(s[1]) + " × " + fmt(s[2]) + " " + t("mm"));
      row("center", [0, 1, 2].map(function (i) { return fmt((mn[i] + mx[i]) / 2); }).join(", ") + " " + t("mm"));
    } else {
      row("position_mm", d.position_mm);
    }
    Object.keys(d.facts || {}).forEach(function (k) { row(k, k === "absorbing_extent" ? t("absorb_" + d.facts[k]) : valueText(d.facts[k])); });
    details.appendChild(table);
    if (hit.kind === "part" && d.surfaces && d.surfaces.length) this.surfacePanel(details, hit);
    this.requestRender();
  };

  Viewer.prototype.surfaceFacts = function (partDoc, surfaceId) {
    if (!surfaceId || !partDoc.surfaces) return null;
    return partDoc.surfaces.filter(function (s) { return s.surface_id === surfaceId; })[0] || null;
  };

  Viewer.prototype.surfaceLine = function (s) {
    var shape = s.sag ? " · " + s.sag.split(":")[0] : "";  // a two-variable sag shape: its kind and normalization, the terms in the facts below
    if (s.radius_mm === undefined && s.sag) return t("surface") + " " + s.surface_id + " · " + t(s.role) + shape;
    return t("surface") + " " + s.surface_id + " · " + t(s.role) + (s.radius_mm !== undefined ? " · R " + fmt(s.radius_mm) + " mm" : " · R ∞") + shape;
  };

  // The selected part's optical surfaces: a box per surface shows / hides it, a click on its name selects it; the
  // selected surface's published facts (profile, clear aperture, media) follow.
  Viewer.prototype.surfacePanel = function (details, hit) {
    var self = this, d = hit.entry.doc;
    var off = this.hiddenSurfaces[d.id] || (this.hiddenSurfaces[d.id] = {});
    details.appendChild(el("h2", "", t("surfaces")));
    d.surfaces.forEach(function (s) {
      var row = el("div", "legend-row"), box = el("input");
      box.type = "checkbox";
      box.checked = !off[s.surface_id];
      box.addEventListener("change", function () { off[s.surface_id] = !box.checked; self.applyVisibility(); });
      var name = el("span", hit.surface === s.surface_id ? "surface-name selected" : "surface-name", self.surfaceLine(s));
      name.addEventListener("click", function () { self.select({ kind: "part", entry: hit.entry, surface: s.surface_id }); });
      row.append(box, name);
      details.appendChild(row);
    });
    details.appendChild(el("div", "count", t("surfaceHint")));
    var sf = this.surfaceFacts(d, hit.surface);
    if (!sf) return;
    var table = el("table");
    Object.keys(sf).forEach(function (k) {
      var v = sf[k];
      if (v === undefined || v === null) return;
      var text = Array.isArray(v) ? v.map(function (x) { return typeof x === "number" ? fmt(x) : x; }).join(", ") : (k === "role" ? t(v) : valueText(v));
      var tr = el("tr");
      tr.append(el("td", "", t(k === "surface_id" ? "surface" : k)), el("td", "", text));
      table.appendChild(tr);
    });
    details.appendChild(table);
  };

  // ------------------------------------------------------------------------------------------------ render loop

  Viewer.prototype.resize = function () {
    var stage = $("stage"), w = Math.max(stage.clientWidth, 1), h = Math.max(stage.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.width = w;
    this.height = h;
    this.lineMaterials.forEach(function (m) { m.resolution.set(w, h); });
    this.perspective.aspect = w / h;
    this.perspective.updateProjectionMatrix();
    if (this.orthographic.top > 0 && !this.autoFrame) {  // keep the scale, follow the aspect
      this.orthographic.left = -this.orthographic.top * w / h;
      this.orthographic.right = this.orthographic.top * w / h;
      this.orthographic.updateProjectionMatrix();
    }
    if (this.autoFrame) {
      var f = this.autoFrame;
      this.frame(f.dir, f.up, f.box, f.aim);
    }
    this.renderNow();
  };

  Viewer.prototype.requestRender = function () {
    var self = this;
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(function () { self.pending = false; self.renderNow(); });
  };

  Viewer.prototype.renderNow = function () {
    this.camera.updateMatrixWorld();
    this.updateOutlines();
    this.renderer.render(this.scene, this.camera);
    this.updateLabels();
    this.updateTriad();
  };

  // The drawing's backdrop: a vertical gradient (sRGB colours, top to bottom), row 0 of the texture at the bottom.
  function backdropTexture(backdrop) {
    var rows = 256, data = new Uint8Array(rows * 4);
    for (var i = 0; i < rows; i++) {
      var k = i / (rows - 1);
      for (var j = 0; j < 3; j++) data[i * 4 + j] = Math.round(255 * (backdrop.bottom[j] * (1 - k) + backdrop.top[j] * k));
      data[i * 4 + 3] = 255;
    }
    var texture = new T.DataTexture(data, 1, rows);
    texture.colorSpace = T.SRGBColorSpace;
    texture.magFilter = T.LinearFilter;
    texture.minFilter = T.LinearFilter;
    texture.needsUpdate = true;
    return texture;
  }

  function fail(message) {
    var main = $("main");
    main.textContent = "";
    main.appendChild(el("div", "", message));
    main.firstChild.id = "error";
  }

  loadDocument().then(function (doc) {
    try {
      var viewer = new Viewer(doc);
      viewer.init();
      window.__opensciViewer = viewer;
    } catch (e) {
      fail(t("webgl") + " " + (e && e.message ? e.message : ""));
    }
  }, function (e) {
    fail(t("loadError") + " " + (e && e.message ? e.message : ""));
  });
})();
