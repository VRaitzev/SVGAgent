import React, { useEffect, useRef, useState } from "react";
import Moveable from "react-moveable";
import {
  Undo2,
  Redo2,
  Download,
  Sparkles,
  Layers,
  Grid as GridIcon,
  Search,
  Plus,
  Trash2,
  Eye,
  EyeOff,
  Copy,
  Send,
  History,
  Palette as PaletteIcon,
  AlertTriangle,
  CheckCircle2,
  Maximize2,
  Minimize2,
  RefreshCw,
} from "lucide-react";
import { api } from "./api";

export default function App() {
  const [scene, setScene] = useState(null);
  const [library, setLibrary] = useState([]);
  const [palettes, setPalettes] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const [selectedId, setSelectedId] = useState(null);
  const [activeLayer, setActiveLayer] = useState("Main");
  const [showGrid, setShowGrid] = useState(true);
  const [strictValidation, setStrictValidation] = useState(true);
  const [zoom, setZoom] = useState(0.68);

  const [liveTransform, setLiveTransform] = useState({});
  const [targetEl, setTargetEl] = useState(null);

  const [chatInput, setChatInput] = useState("");
  const [contextMode, setContextMode] = useState("multimodal");
  const [libraryMode, setLibraryMode] = useState("full_catalog"); // full_catalog | rag
  const [ragTopK, setRagTopK] = useState(10);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text: "Привет! Я AI-дизайнер Antigravity. Учтены все правила ТЗ и референсы Figma: единые пропорции объектов, переключение Главный ↔ Фоновый, плотная изометрическая компоновка с наслоением и обрезкой краями, 3+ палитры и выбор режима библиотеки (весь каталог или RAG Top-K).",
    },
  ]);

  const [toast, setToast] = useState(null);
  const itemRefs = useRef({});

  const showToast = (message, type = "error") => {
    setToast({ message, type });
    setTimeout(() => {
      setToast((prev) => (prev?.message === message ? null : prev));
    }, 5500);
  };

  const refreshLibrary = async (q = searchQuery, level = scene?.detail_level || "L", rFilter = roleFilter) => {
    try {
      const results = await api.searchLibrary(q, level, rFilter === "all" ? "" : rFilter);
      setLibrary(results);
    } catch (err) {
      showToast(err.message);
    }
  };

  useEffect(() => {
    async function init() {
      try {
        const [sceneData, palData] = await Promise.all([api.getScene(), api.getPalettes()]);
        setScene(sceneData);
        setPalettes(palData);
        const initialLevel = sceneData?.detail_level || "L";
        const libData = await api.searchLibrary("", initialLevel, "");
        setLibrary(libData);
        if (sceneData?.layers?.length) {
          setActiveLayer(sceneData.layers[sceneData.layers.length - 1].name);
        }
      } catch (err) {
        showToast(`Ошибка подключения к серверу: ${err.message}`);
      }
    }
    init();
  }, []);

  useEffect(() => {
    if (selectedId && itemRefs.current[selectedId]) {
      setTargetEl(itemRefs.current[selectedId]);
    } else {
      setTargetEl(null);
    }
  }, [selectedId, scene]);

  const runDiff = async (operations, commitMessage) => {
    try {
      const res = await api.applyDiff(operations, commitMessage, strictValidation);
      setScene(res.scene);
      setLiveTransform({});
      if (operations.some((o) => o.op === "set_detail_level")) {
        await refreshLibrary(searchQuery, res.scene.detail_level, roleFilter);
      }
      return true;
    } catch (err) {
      setLiveTransform({});
      showToast(err.message, "error");
      return false;
    }
  };

  const handleSwitchDetailLevel = async (nextLevel) => {
    if (!scene || scene.detail_level === nextLevel) return;
    await runDiff(
      [{ op: "set_detail_level", params: { detail_level: nextLevel } }],
      `Переключение детализации сцены на размер ${nextLevel}`
    );
  };

  const handleAddFromLibrary = async (component) => {
    const canvasW = scene?.canvas?.width || 800;
    const canvasH = scene?.canvas?.height || 800;
    const scale = scene?.uniform_scale || 1.0;
    const x = Math.round((canvasW - component.width * scale) / 2);
    const y = Math.round((canvasH - component.height * scale) / 2);
    const targetLayer = component.role === "background" ? "Background" : "Main";
    setActiveLayer(targetLayer);

    await runDiff(
      [
        {
          op: "add_item",
          params: {
            image_id: component.id,
            x,
            y,
            rotation: 0.0,
            layer: targetLayer,
          },
        },
      ],
      `Добавлен "${component.name}" (${component.role}, ${component.detail_level})`
    );
  };

  const handleAddCustomPalette = async () => {
    const title = prompt("Название новой цветовой схемы:", "Изумрудная схема");
    if (!title) return;
    const bgStr = prompt("Цвета фонов через запятую (HEX):", "#a7f3d0, #6ee7b7, #d1fae5");
    const accStr = prompt("Акцентные цвета главных объектов через запятую (HEX):", "#059669, #fde68a, #f43f5e");
    const canvasBg = prompt("Цвет подложки холста (HEX):", "#d1fae5") || "#ffffff";
    if (!bgStr || !accStr) return;

    const name = `custom_${Date.now()}`;
    try {
      await api.registerPalette({
        name,
        title,
        bg_colors: bgStr.split(",").map((s) => s.trim()).filter(Boolean),
        accent_colors: accStr.split(",").map((s) => s.trim()).filter(Boolean),
        canvas_bg: canvasBg.trim(),
      });
      const updatedPalettes = await api.getPalettes();
      setPalettes(updatedPalettes);
      await runDiff([{ op: "apply_palette", params: { palette_name: name } }], `Применена схема "${title}"`);
    } catch (err) {
      showToast(err.message);
    }
  };

  const handleUndo = async () => {
    try {
      const updated = await api.undo();
      setScene(updated);
      setLiveTransform({});
      await refreshLibrary(searchQuery, updated.detail_level, roleFilter);
    } catch (err) {
      showToast(err.message);
    }
  };

  const handleRedo = async () => {
    try {
      const updated = await api.redo();
      setScene(updated);
      setLiveTransform({});
      await refreshLibrary(searchQuery, updated.detail_level, roleFilter);
    } catch (err) {
      showToast(err.message);
    }
  };

  const handleRevert = async (stepId) => {
    try {
      const updated = await api.revert(stepId);
      setScene(updated);
      setLiveTransform({});
      await refreshLibrary(searchQuery, updated.detail_level, roleFilter);
    } catch (err) {
      showToast(err.message);
    }
  };

  const handleSendChat = async (e) => {
    e.preventDefault();
    const promptText = chatInput.trim();
    if (!promptText || chatLoading) return;

    setChatMessages((prev) => [...prev, { role: "user", text: promptText }]);
    setChatInput("");
    setChatLoading(true);

    try {
      const res = await api.chat(promptText, contextMode, libraryMode, ragTopK);
      setScene(res.scene);
      await refreshLibrary(searchQuery, res.scene.detail_level, roleFilter);

      const safeAnswer =
        typeof res.answer === "string"
          ? res.answer
          : Array.isArray(res.answer)
          ? res.answer.filter((b) => b?.type === "text").map((b) => b.text).join("\n")
          : JSON.stringify(res.answer);

      setChatMessages((prev) => [...prev, { role: "assistant", text: safeAnswer }]);
    } catch (err) {
      setChatMessages((prev) => [...prev, { role: "assistant", text: `⚠️ Ошибка: ${err.message}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  const selectedItem = scene?.items?.find((it) => it.id === selectedId) || null;

  const getItemDisplayProps = (item) => {
    const live = liveTransform[item.id];
    return {
      x: live?.x ?? item.x,
      y: live?.y ?? item.y,
      scale: live?.scale ?? item.scale,
      rotation: live?.rotation ?? item.rotation,
    };
  };

  if (!scene) {
    return (
      <div className="loading-screen">
        <Sparkles size={32} className="spin" />
        <p>Загрузка студии Antigravity...</p>
      </div>
    );
  }

  const { width: canvasW, height: canvasH, background_color: canvasBg, show_isometric_guides: showIsoGuides } = scene.canvas;
  const { columns: gridCols, rows: gridRows } = scene.grid;
  const activePalette = scene.active_palette;

  const allowedSwatchesForSelected =
    selectedItem && activePalette
      ? selectedItem.role === "background" || (selectedItem.layer_name || "").toLowerCase() === "background"
        ? Array.from(new Set([...(activePalette.bg_colors || []), activePalette.canvas_bg].filter(Boolean)))
        : Array.from(
            new Set([
              ...(activePalette.accent_colors || []),
              ...(activePalette.bg_colors || []),
              activePalette.canvas_bg,
            ].filter(Boolean))
          )
      : [];

  return (
    <div className="studio-layout">
      {/* ВЕРХНЯЯ ПАНЕЛЬ */}
      <header className="topbar">
        <div className="brand">
          <Sparkles size={20} className="brand-icon" />
          <span>Antigravity</span>
          <span className="badge">Шаг #{scene.step_id}</span>
          {scene.validation?.ok ? (
            <span className="status-pill valid">
              <CheckCircle2 size={14} /> Валидна
            </span>
          ) : (
            <span className="status-pill invalid">
              <AlertTriangle size={14} /> Ошибка правил
            </span>
          )}
        </div>

        <div className="toolbar-group">
          <button className="btn-icon" onClick={handleUndo} title="Отменить (Undo)">
            <Undo2 size={17} />
          </button>
          <button className="btn-icon" onClick={handleRedo} title="Повторить (Redo)">
            <Redo2 size={17} />
          </button>

          <div className="divider" />

          {/* ПЕРЕКЛЮЧАТЕЛЬ ДЕТАЛИЗАЦИИ L / S */}
          <div className="segmented-control" title="Уровень детализации иллюстраций (автоматически подменяет элементы на холсте)">
            <button
              className={`seg-btn ${scene.detail_level === "L" ? "active" : ""}`}
              onClick={() => handleSwitchDetailLevel("L")}
            >
              <Maximize2 size={13} /> Размер L
            </button>
            <button
              className={`seg-btn ${scene.detail_level === "S" ? "active" : ""}`}
              onClick={() => handleSwitchDetailLevel("S")}
            >
              <Minimize2 size={13} /> Размер S
            </button>
          </div>

          <div className="divider" />

          {/* ЦВЕТОВАЯ СХЕМА */}
          <PaletteIcon size={15} className="muted-icon" />
          <select
            className="select-input"
            value={activePalette?.name || ""}
            onChange={(e) => {
              if (e.target.value) {
                runDiff(
                  [{ op: "apply_palette", params: { palette_name: e.target.value } }],
                  `Применена цветовая схема ${e.target.value}`
                );
              }
            }}
          >
            <option value="">Цветовая схема...</option>
            {palettes.map((p) => (
              <option key={p.id || p.name} value={p.name}>
                {p.title || p.name}
              </option>
            ))}
          </select>
          <button className="btn-xs" onClick={handleAddCustomPalette} title="Добавить новую цветовую схему">
            <Plus size={13} /> Схема
          </button>

          <div className="divider" />

          {/* ЕДИНЫЙ МАСШТАБ СЦЕНЫ */}
          <div className="zoom-controls" title="Единый относительный масштаб всех объектов сцены">
            <span>Масштаб:</span>
            <input
              type="number"
              step="0.1"
              min="0.3"
              max="2.5"
              className="scale-mini-input"
              value={scene.uniform_scale ?? 1.0}
              onChange={(e) => {
                const val = Number(e.target.value);
                if (val > 0) {
                  runDiff([{ op: "set_uniform_scale", params: { scale: val } }], `Единый масштаб сцены ${val}`);
                }
              }}
            />
          </div>

          <button
            className={`btn-toggle ${showGrid ? "active" : ""}`}
            onClick={() => setShowGrid(!showGrid)}
          >
            <GridIcon size={15} /> {gridCols}×{gridRows}
          </button>

          <label className="checkbox-pill" title="Строгая проверка правил ТЗ">
            <input
              type="checkbox"
              checked={strictValidation}
              onChange={(e) => setStrictValidation(e.target.checked)}
            />
            Валидация
          </label>

          <div className="zoom-controls">
            <button className="btn-sm" onClick={() => setZoom((z) => Math.max(0.3, +(z - 0.1).toFixed(2)))}>
              -
            </button>
            <span>{Math.round(zoom * 100)}%</span>
            <button className="btn-sm" onClick={() => setZoom((z) => Math.min(1.3, +(z + 0.1).toFixed(2)))}>
              +
            </button>
          </div>
        </div>

        <div className="export-group">
          <a className="btn-secondary" href={api.exportUrl("svg")} download="antigravity_scene.svg">
            <Download size={15} /> SVG
          </a>
          <a className="btn-primary" href={api.exportUrl("png")} download="antigravity_scene.png">
            <Download size={15} /> PNG
          </a>
        </div>
      </header>

      {toast && (
        <div className={`toast-banner ${toast.type}`} onClick={() => setToast(null)}>
          <AlertTriangle size={18} />
          <pre>{toast.message}</pre>
        </div>
      )}

      <div className="workspace">
        {/* ЛЕВАЯ ПАНЕЛЬ */}
        <aside className="sidebar-left">
          <div className="panel-section">
            <div className="section-header">
              <span>Библиотека ({scene.detail_level})</span>
              <span className="count-tag">{library.length}</span>
            </div>

            <div className="role-tabs">
              {[
                { key: "all", label: "Все" },
                { key: "main", label: "Главные" },
                { key: "background", label: "Фоновые" },
              ].map((tab) => (
                <button
                  key={tab.key}
                  className={`role-tab ${roleFilter === tab.key ? "active" : ""}`}
                  onClick={() => {
                    setRoleFilter(tab.key);
                    refreshLibrary(searchQuery, scene.detail_level, tab.key);
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <form
              className="search-bar"
              onSubmit={(e) => {
                e.preventDefault();
                refreshLibrary(searchQuery, scene.detail_level, roleFilter);
              }}
            >
              <Search size={15} />
              <input
                type="text"
                placeholder="Поиск по метафоре..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </form>

            <div className="library-grid">
              {library.map((comp) => (
                <div
                  key={comp.id}
                  className="library-card"
                  onClick={() => handleAddFromLibrary(comp)}
                  title={`${comp.name} [${comp.role === "background" ? "Фоновый монохром" : "Главный цветной"} · ${comp.detail_level} · ${Math.round(comp.width)}×${Math.round(comp.height)}px]
${comp.description || ""}`}
                >
                  <div className="card-badges">
                    <span className={`role-badge ${comp.role}`}>
                      {comp.role === "background" ? "Фон" : "Главный"}
                    </span>
                    <span className="size-badge">{comp.detail_level}</span>
                  </div>
                  <div className="svg-thumb" dangerouslySetInnerHTML={{ __html: comp.svg }} />
                  <div className="card-title">{comp.name}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel-section layers-section">
            <div className="section-header">
              <span>
                <Layers size={14} /> Слои сцены
              </span>
            </div>

            <div className="layers-list">
              {[...scene.layers].reverse().map((layer) => (
                <div
                  key={layer.id}
                  className={`layer-group ${activeLayer === layer.name ? "active-layer" : ""}`}
                >
                  <div className="layer-header" onClick={() => setActiveLayer(layer.name)}>
                    <button
                      className="btn-ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        runDiff(
                          [{ op: "update_layer", params: { layer: layer.id, visible: !layer.visible } }],
                          `Видимость слоя ${layer.name}`
                        );
                      }}
                    >
                      {layer.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                    </button>
                    <span className="layer-title">{layer.name}</span>
                    <span className="layer-meta">
                      {layer.name.toLowerCase() === "background" ? "монохром · опц." : "главные"}
                    </span>
                  </div>

                  <div className="layer-items">
                    {[...layer.items].reverse().map((item) => (
                      <div
                        key={item.id}
                        className={`layer-item ${selectedId === item.id ? "selected" : ""}`}
                        onClick={() => setSelectedId(item.id)}
                      >
                        <span>{item.name || item.image_id.slice(0, 6)}</span>
                        <span className="coords-hint">
                          {item.detail_level} · {Math.round(item.x)},{Math.round(item.y)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* ЦЕНТРАЛЬНЫЙ ХОЛСТ */}
        <main
          className="canvas-viewport"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget || e.target.classList.contains("artboard")) {
              setSelectedId(null);
            }
          }}
        >
          <div
            className="artboard"
            style={{
              width: canvasW,
              height: canvasH,
              backgroundColor: canvasBg || "#ffffff",
              transform: `scale(${zoom})`,
              transformOrigin: "center center",
              overflow: "hidden",
            }}
          >
            {showIsoGuides && (
              <svg
                className="iso-guides-svg"
                width={canvasW}
                height={canvasH}
                viewBox={`0 0 ${canvasW} ${canvasH}`}
              >
                <line x1="0" y1={canvasH * 0.45} x2={canvasW * 0.78} y2="0" stroke="#1e1b4b" strokeWidth="1.5" />
                <line x1={canvasW * 0.25} y1={canvasH} x2={canvasW} y2={canvasH * 0.56} stroke="#1e1b4b" strokeWidth="1.5" />
                <line x1={canvasW * 0.55} y1="0" x2={canvasW} y2={canvasH * 0.26} stroke="#1e1b4b" strokeWidth="1.5" />
              </svg>
            )}

            {showGrid && (
              <div
                className="canvas-grid-overlay"
                style={{
                  gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
                  gridTemplateRows: `repeat(${gridRows}, 1fr)`,
                }}
              >
                {Array.from({ length: gridCols * gridRows }).map((_, idx) => {
                  const c = idx % gridCols;
                  const r = Math.floor(idx / gridCols);
                  return (
                    <div key={idx} className="grid-cell">
                      <span className="cell-label">
                        [{c}, {r}]
                      </span>
                    </div>
                  );
                })}
              </div>
            )}

            {scene.items.map((item) => {
              const disp = getItemDisplayProps(item);
              const w = item.original_width * disp.scale;
              const h = item.original_height * disp.scale;
              const isDragging = Boolean(liveTransform[item.id]);

              return (
                <div
                  key={item.id}
                  ref={(el) => {
                    if (el) itemRefs.current[item.id] = el;
                  }}
                  className={`scene-item ${selectedId === item.id ? "is-selected" : ""} ${
                    isDragging ? "is-dragging" : "smooth-transition"
                  }`}
                  style={{
                    width: `${w}px`,
                    height: `${h}px`,
                    transform: `translate(${disp.x}px, ${disp.y}px) rotate(${disp.rotation}deg)`,
                  }}
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    setSelectedId(item.id);
                    setActiveLayer(item.layer_name);
                  }}
                >
                  <div className="scene-item-svg" dangerouslySetInnerHTML={{ __html: item.svg }} />
                </div>
              );
            })}

            {selectedItem && targetEl && (
              <Moveable
                target={targetEl}
                draggable={true}
                scalable={true}
                rotatable={true}
                keepRatio={true}
                throttleDrag={1}
                throttleRotate={1}
                onDrag={({ beforeTranslate }) => {
                  setLiveTransform((prev) => ({
                    ...prev,
                    [selectedItem.id]: {
                      ...(prev[selectedItem.id] || selectedItem),
                      x: beforeTranslate[0],
                      y: beforeTranslate[1],
                    },
                  }));
                }}
                onDragEnd={({ lastEvent }) => {
                  if (!lastEvent) return;
                  const [nx, ny] = lastEvent.beforeTranslate;
                  runDiff(
                    [
                      {
                        op: "update_item",
                        params: {
                          item_id: selectedItem.id,
                          x: Math.round(nx * 10) / 10,
                          y: Math.round(ny * 10) / 10,
                        },
                      },
                    ],
                    `Перемещение "${selectedItem.name}"`
                  );
                }}
                onScale={({ scale, drag }) => {
                  const baseScale = selectedItem.scale;
                  const nextScale = Math.max(0.2, +(baseScale * scale[0]).toFixed(3));
                  setLiveTransform((prev) => ({
                    ...prev,
                    [selectedItem.id]: {
                      ...(prev[selectedItem.id] || selectedItem),
                      scale: nextScale,
                      x: drag.beforeTranslate[0],
                      y: drag.beforeTranslate[1],
                    },
                  }));
                }}
                onScaleEnd={({ lastEvent }) => {
                  if (!lastEvent) return;
                  const live = liveTransform[selectedItem.id];
                  if (!live) return;
                  runDiff(
                    [{ op: "set_uniform_scale", params: { scale: live.scale } }],
                    `Единый масштаб сцены (${live.scale})`
                  );
                }}
                onRotate={({ beforeRotate }) => {
                  setLiveTransform((prev) => ({
                    ...prev,
                    [selectedItem.id]: {
                      ...(prev[selectedItem.id] || selectedItem),
                      rotation: Math.round(beforeRotate * 10) / 10,
                    },
                  }));
                }}
                onRotateEnd={({ lastEvent }) => {
                  if (!lastEvent) return;
                  const rot = Math.round(lastEvent.beforeRotate * 10) / 10;
                  runDiff(
                    [{ op: "update_item", params: { item_id: selectedItem.id, rotation: rot } }],
                    `Поворот "${selectedItem.name}" (${rot}°)`
                  );
                }}
              />
            )}
          </div>
        </main>

        {/* ПРАВАЯ ПАНЕЛЬ */}
        <aside className="sidebar-right">
          <div className="panel-section inspector-section">
            <div className="section-header">
              <span>Инспектор объекта</span>
              {selectedItem && (
                <div className="inspector-actions">
                  <button
                    className="btn-xs"
                    title="Переключить состояние: Главный ↔ Фоновый"
                    onClick={() =>
                      runDiff(
                        [{ op: "toggle_item_role", params: { item_id: selectedItem.id } }],
                        `Смена роли "${selectedItem.name}"`
                      )
                    }
                  >
                    <RefreshCw size={12} /> {selectedItem.role === "background" ? "В Главный" : "В Фоновый"}
                  </button>
                  <button
                    className="btn-xs"
                    title="Дублировать"
                    onClick={() =>
                      runDiff(
                        [{ op: "duplicate_item", params: { item_id: selectedItem.id, dx: 40, dy: 40 } }],
                        `Дубликат ${selectedItem.name}`
                      )
                    }
                  >
                    <Copy size={13} />
                  </button>
                  <button
                    className="btn-xs danger"
                    title="Удалить"
                    onClick={() => {
                      const idToRemove = selectedItem.id;
                      setSelectedId(null);
                      runDiff([{ op: "remove_item", params: { item_id: idToRemove } }], `Удалён объект`);
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              )}
            </div>

            {selectedItem ? (
              <div className="inspector-form">
                <div className="prop-row">
                  <label>Состояние</label>
                  <span className="role-info-pill">
                    {selectedItem.role === "background" ? "Фоновый (монохром)" : "Главный (цветной)"} ·{" "}
                    {selectedItem.layer_name}
                  </span>
                </div>

                <div className="prop-grid-2">
                  <div>
                    <label>X (px)</label>
                    <input
                      type="number"
                      value={Math.round(selectedItem.x)}
                      onChange={(e) =>
                        runDiff(
                          [{ op: "update_item", params: { item_id: selectedItem.id, x: Number(e.target.value) } }],
                          "Изменение X"
                        )
                      }
                    />
                  </div>
                  <div>
                    <label>Y (px)</label>
                    <input
                      type="number"
                      value={Math.round(selectedItem.y)}
                      onChange={(e) =>
                        runDiff(
                          [{ op: "update_item", params: { item_id: selectedItem.id, y: Number(e.target.value) } }],
                          "Изменение Y"
                        )
                      }
                    />
                  </div>
                </div>

                {(selectedItem.original_colors || []).length > 0 && (allowedSwatchesForSelected || []).length > 0 && (
                  <div className="color-swatches">
                    <label>
                      Цвета схемы ({selectedItem.role === "background" ? "только фоновые" : "акцентные + фоновые"}):
                    </label>
                    {selectedItem.original_colors.map((origColor) => {
                      const currentVal = selectedItem.colors?.[origColor] || origColor;
                      return (
                        <div key={origColor} className="swatch-group-row">
                          <span className="orig-dot" style={{ background: origColor }} title={`Исходный: ${origColor}`} />
                          <span>→</span>
                          <div className="swatch-row">
                            {allowedSwatchesForSelected.map((palColor) => (
                              <button
                                key={palColor}
                                type="button"
                                className={`palette-dot ${currentVal === palColor ? "active" : ""}`}
                                style={{ background: palColor }}
                                title={palColor}
                                onClick={() => {
                                  const nextColors = {
                                    ...(selectedItem.colors || {}),
                                    [origColor]: palColor,
                                  };
                                  runDiff(
                                    [{ op: "update_item", params: { item_id: selectedItem.id, colors: nextColors } }],
                                    `Перекраска ${selectedItem.name}`
                                  );
                                }}
                              />
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : (
              <p className="empty-hint">
                Выберите объект на холсте, чтобы переключить его состояние (Главный ↔ Фоновый), сдвинуть или перекрасить.
              </p>
            )}
          </div>

          <div className="panel-section chat-section">
            <div className="section-header">
              <span>
                <Sparkles size={14} /> AI-Агент
              </span>
              <div className="agent-controls-inline">
                <select
                  className="mode-select"
                  value={libraryMode}
                  onChange={(e) => setLibraryMode(e.target.value)}
                  title="Как агент получает доступ к библиотеке SVG"
                >
                  <option value="full_catalog">Весь каталог</option>
                  <option value="rag">RAG-поиск</option>
                </select>
                {libraryMode === "rag" && (
                  <select
                    className="mode-select"
                    value={ragTopK}
                    onChange={(e) => setRagTopK(Number(e.target.value))}
                    title="Количество кандидатов из RAG"
                  >
                    <option value={5}>Top 5</option>
                    <option value={10}>Top 10</option>
                    <option value={15}>Top 15</option>
                    <option value={25}>Top 25</option>
                  </select>
                )}
                <select
                  className="mode-select"
                  value={contextMode}
                  onChange={(e) => setContextMode(e.target.value)}
                >
                  <option value="text_only">Text</option>
                  <option value="multimodal">VLM</option>
                </select>
              </div>
            </div>

            <div className="chat-messages">
              {chatMessages.map((m, idx) => (
                <div key={idx} className={`chat-bubble ${m.role}`}>
                  {typeof m.text === "string" ? m.text : JSON.stringify(m.text)}
                </div>
              ))}
              {chatLoading && (
                <div className="chat-bubble assistant thinking">
                  ⏳ Агент собирает изометрическую композицию...
                </div>
              )}
            </div>

            <form className="chat-input-row" onSubmit={handleSendChat}>
              <input
                type="text"
                placeholder="Например: Собери сцену про учёбу с книгой, лампой и ручкой..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={chatLoading}
              />
              <button type="submit" className="btn-primary" disabled={chatLoading}>
                <Send size={15} />
              </button>
            </form>
          </div>

          <div className="panel-section history-section">
            <div className="section-header">
              <span>
                <History size={14} /> История транзакций
              </span>
            </div>
            <div className="history-list">
              {[...(scene.history || [])].reverse().map((h) => (
                <div
                  key={h.step_id}
                  className={`history-item ${h.is_current ? "current" : ""}`}
                  onClick={() => handleRevert(h.step_id)}
                >
                  <span className="step-num">#{h.step_id}</span>
                  <span className="step-msg">{h.message}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}