"use strict";

// ------------------------------------------------------------------------ //
// Estado
// ------------------------------------------------------------------------ //
const estado = {
  dados: null,          // resposta de /api/analisar
  base: null,           // Image da radiografia
  mapas: {},            // chave -> Image do mapa de calor (RGBA)
  visiveis: new Set(),  // chaves de achados exibidos
  destaque: null,       // chave em destaque (passar o mouse / clique)
  camadas: { mapas: true, contornos: true, rotulos: true, ict: true, anatomia: false },
  opacidade: 0.55, brilho: 1, contraste: 1, inverter: false,
  zoom: 1, panX: 0, panY: 0,
  arquivo: null,
};

const $ = (id) => document.getElementById(id);
const tela = $("tela");
const ctx = tela.getContext("2d");
const area = $("area");

const STATUS = { positivo: "Positivo", indeterminado: "Indeterminado", negativo: "Negativo" };
const pct = (v) => `${Math.round(v * 100)}%`;
const capitalizar = (t) => (t ? t.charAt(0).toUpperCase() + t.slice(1) : t);

function el(tag, attrs = {}, ...filhos) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const f of filhos.flat()) {
    if (f === null || f === undefined || f === false) continue;
    n.append(f instanceof Node ? f : document.createTextNode(String(f)));
  }
  return n;
}

function carregarImagem(src) {
  return new Promise((ok, falha) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = falha;
    img.src = src;
  });
}

// ------------------------------------------------------------------------ //
// Envio do arquivo
// ------------------------------------------------------------------------ //
async function analisar(arquivo, forcar = false) {
  estado.arquivo = arquivo;
  esconderErro();
  $("carregando").hidden = false;
  $("soltar").hidden = true;
  const form = new FormData();
  form.append("arquivo", arquivo, arquivo.name);
  form.append("forcar", forcar ? "true" : "false");
  form.append("anonimizar", $("opt-anonimizar").checked ? "true" : "false");
  try {
    const resp = await fetch("/api/analisar", { method: "POST", body: form });
    const corpo = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const mensagem = corpo.detail || `Falha na análise (HTTP ${resp.status}).`;
      mostrarErro(mensagem, resp.status === 422);
      if (!estado.dados) $("soltar").hidden = false;
      return;
    }
    await exibirResultado(corpo);
  } catch (e) {
    mostrarErro(`Não foi possível contatar o servidor: ${e.message}`);
    if (!estado.dados) $("soltar").hidden = false;
  } finally {
    $("carregando").hidden = true;
  }
}

function mostrarErro(texto, permitirForcar = false) {
  $("erro-texto").textContent = texto;
  $("btn-forcar").hidden = !permitirForcar;
  $("erro").hidden = false;
}
function esconderErro() { $("erro").hidden = true; }

// ------------------------------------------------------------------------ //
// Exibição do resultado
// ------------------------------------------------------------------------ //
async function exibirResultado(dados) {
  estado.dados = dados;
  estado.base = await carregarImagem(dados.imagem);
  estado.mapas = {};
  await Promise.all(dados.achados.filter((a) => a.mapa).map(async (a) => {
    estado.mapas[a.chave] = await carregarImagem(a.mapa);
  }));
  estado.visiveis = new Set(dados.achados.filter((a) => a.status !== "negativo" && !a.suprimido).map((a) => a.chave));
  estado.destaque = null;
  $("vazio").hidden = true;
  ajustarATela();
  preencherAchados();
  preencherLaudo(dados.laudo);
  preencherEscores();
  preencherExame();
  const ativa = document.querySelector('.abas [aria-selected="true"]').dataset.aba;
  selecionarAba(ativa);
}

function achadosRelevantes() {
  const ordem = { positivo: 0, indeterminado: 1 };
  return estado.dados.achados
    .filter((a) => a.status !== "negativo")
    .sort((a, b) => (a.suprimido - b.suprimido) || (ordem[a.status] - ordem[b.status]) || (b.escore - a.escore));
}

function preencherAchados() {
  const { laudo, avisos } = estado.dados;
  const urgente = estado.dados.achados.some((a) => a.status === "positivo" && a.gravidade >= 3);
  const caixa = $("impressao");
  caixa.className = "impressao" + (urgente ? " urgente" : "");
  caixa.replaceChildren(el("h3", {}, "Impressão diagnóstica"), el("ol", {}, laudo.impressao.map((t) => el("li", {}, t))));

  $("avisos-resumo").replaceChildren(...(avisos.length
    ? [el("div", { class: "aviso-curto" }, `${avisos.length} observação(ões) técnica(s) — veja a aba Exame.`)]
    : []));

  const lista = $("lista-achados");
  const relevantes = achadosRelevantes();
  lista.replaceChildren();
  if (!relevantes.length) {
    lista.append(el("p", { class: "nota" }, "Nenhum achado acima dos limiares de detecção."));
  }
  relevantes.forEach((a) => {
    const marcado = estado.visiveis.has(a.chave);
    const caixaSel = el("input", { type: "checkbox", checked: marcado, "aria-label": `Exibir ${a.nome}` });
    caixaSel.addEventListener("click", (ev) => ev.stopPropagation());
    caixaSel.addEventListener("change", () => {
      if (caixaSel.checked) estado.visiveis.add(a.chave); else estado.visiveis.delete(a.chave);
      cartao.classList.toggle("oculto", !caixaSel.checked);
      desenhar();
    });
    const cartao = el("div", { class: "cartao" + (marcado ? "" : " oculto"), "data-chave": a.chave },
      el("span", { class: "cor", style: `background:${a.cor}` }),
      el("div", {},
        el("span", { class: "nome" }, a.nome),
        el("span", { class: `etiqueta ${a.status}` }, STATUS[a.status]),
        a.suprimido ? el("span", { class: "etiqueta redundante", title: "Incluído em achado mais específico" }, "redundante") : null),
      el("div", { style: "display:flex;gap:8px;align-items:center" },
        el("span", { class: "escore-num" }, pct(a.escore)),
        el("label", {}, caixaSel)),
      el("div", { class: "barra" }, el("i", { style: `width:${pct(a.escore)};background:${a.cor}` })),
      a.local ? el("div", { class: "local" }, capitalizar(a.local)) : null,
    );
    cartao.addEventListener("mouseenter", () => { estado.destaque = a.chave; desenhar(); });
    cartao.addEventListener("mouseleave", () => { estado.destaque = null; desenhar(); });
    cartao.addEventListener("click", () => focarAchado(a.chave));
    lista.append(cartao);
  });

  const negativos = estado.dados.achados.filter((a) => a.status === "negativo");
  $("n-negativos").textContent = negativos.length;
  $("lista-negativos").replaceChildren(...negativos.map((a) =>
    el("div", { class: "linha-negativa" }, el("span", {}, a.nome), el("span", { class: "escore-num" }, pct(a.escore)))));
}

function focarAchado(chave) {
  const a = estado.dados.achados.find((x) => x.chave === chave);
  if (!a || !a.regioes.length) return;
  estado.visiveis.add(chave);
  const xs = [], ys = [];
  a.regioes.forEach((r) => { const [x, y, w, h] = r.caixa; xs.push(x, x + w); ys.push(y, y + h); });
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const largura = tela.clientWidth, altura = tela.clientHeight;
  const margem = 1.8;
  const [W, H] = estado.dados.dimensoes_exibicao;
  const zoomAjuste = Math.min(largura / W, altura / H);
  estado.zoom = Math.max(zoomAjuste, Math.min(largura / ((x1 - x0) * margem), altura / ((y1 - y0) * margem), zoomAjuste * 4));
  estado.panX = largura / 2 - ((x0 + x1) / 2) * estado.zoom;
  estado.panY = altura / 2 - ((y0 + y1) / 2) * estado.zoom;
  document.querySelectorAll(".cartao").forEach((c) => c.classList.toggle("ativo", c.dataset.chave === chave));
  desenhar();
}

// ------------------------------------------------------------------------ //
// Laudo
// ------------------------------------------------------------------------ //
function preencherLaudo(laudo) {
  $("cabecalho-laudo").replaceChildren(...laudo.cabecalho.flatMap(([r, v]) => [el("dt", {}, r), el("dd", {}, v)]));
  $("ed-tecnica").value = laudo.tecnica;
  $("ed-analise").value = laudo.analise.map((s) => `${s.sistema}: ${s.frases.join(" ")}`).join("\n");
  $("ed-indeterminados").value = laudo.achados_indeterminados.join("\n");
  $("ed-medidas").value = laudo.medidas.join("\n");
  $("ed-impressao").value = laudo.impressao.map((t, i) => `${i + 1}. ${t}`).join("\n");
  $("ed-recomendacoes").value = laudo.recomendacoes.join("\n");
  atualizarSelo();
  const id = estado.dados.id;
  $("lnk-png").href = `/api/resultados/${id}/anotada.png`;
  $("lnk-dcm").href = `/api/resultados/${id}/anotada.dcm`;
  $("lnk-json").href = `/api/resultados/${id}/resultado.json`;
}

function edicaoLaudo() {
  return {
    analise: $("ed-analise").value,
    achados_indeterminados: $("ed-indeterminados").value,
    medidas: $("ed-medidas").value,
    impressao: $("ed-impressao").value,
    recomendacoes: $("ed-recomendacoes").value,
    revisor: $("ed-revisor").value,
  };
}

function atualizarSelo() {
  const revisor = $("ed-revisor").value.trim();
  const selo = $("selo-laudo");
  selo.classList.toggle("revisado", !!revisor);
  selo.textContent = revisor ? `Revisado por ${revisor}` : "Pré-laudo gerado por IA — revise e edite antes de emitir";
}

async function baixarComEdicao(caminho, nomePadrao) {
  if (!estado.dados) return;
  const resp = await fetch(`/api/resultados/${estado.dados.id}/${caminho}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(edicaoLaudo()),
  });
  if (!resp.ok) { mostrarErro((await resp.json().catch(() => ({}))).detail || "Falha ao gerar o arquivo."); return; }
  const disposicao = resp.headers.get("Content-Disposition") || "";
  const nome = (disposicao.match(/filename="([^"]+)"/) || [])[1] || nomePadrao;
  const url = URL.createObjectURL(await resp.blob());
  const a = el("a", { href: url, download: nome });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

async function copiarTexto() {
  if (!estado.dados) return;
  const resp = await fetch(`/api/resultados/${estado.dados.id}/laudo.txt`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(edicaoLaudo()),
  });
  const texto = await resp.text();
  try {
    await navigator.clipboard.writeText(texto);
    $("btn-copiar").textContent = "Copiado!";
  } catch {
    const t = el("textarea", { style: "position:fixed;opacity:0" });
    t.value = texto; document.body.append(t); t.select(); document.execCommand("copy"); t.remove();
    $("btn-copiar").textContent = "Copiado!";
  }
  setTimeout(() => { $("btn-copiar").textContent = "Copiar texto"; }, 1600);
}

// ------------------------------------------------------------------------ //
// Escores e dados do exame
// ------------------------------------------------------------------------ //
function preencherEscores() {
  const { achados, limiares } = estado.dados;
  $("lista-escores").replaceChildren(...[...achados].sort((a, b) => b.escore - a.escore).map((a) => {
    const cor = a.status === "negativo" ? "#4b5869" : a.cor;
    return el("div", { class: "linha-escore", title: `${a.nome}: ${STATUS[a.status]}` },
      el("span", {}, a.nome),
      el("div", { class: "trilho" },
        el("i", { style: `width:${pct(a.escore)};background:${cor}` }),
        el("b", { style: `left:${pct(limiares.indeterminado)}` }),
        el("b", { style: `left:${pct(limiares.positivo)}` })),
      el("span", { class: "valor" }, pct(a.escore)));
  }));
}

function preencherExame() {
  const m = estado.dados.metadados;
  const campos = [
    ["Paciente", m.paciente_nome], ["ID", m.paciente_id], ["Sexo", m.sexo], ["Idade", m.idade],
    ["Data", [m.data_exame, m.hora_exame].filter(Boolean).join(" ")], ["Instituição", m.instituicao],
    ["Modalidade", m.modalidade], ["Região", m.regiao], ["Incidência", m.incidencia],
    ["Descrição", m.descricao_estudo], ["Fabricante", m.fabricante],
    ["Dimensões", m.colunas && m.linhas ? `${m.colunas} × ${m.linhas} px` : null],
    ["Espaçamento", m.espacamento_mm ? `${m.espacamento_mm.map((v) => v.toFixed(3)).join(" × ")} mm` : null],
    ["Fotometria", m.fotometria], ["Bits", m.bits], ["Arquivo", estado.dados.arquivo],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "");
  $("dados-exame").replaceChildren(...campos.flatMap(([r, v]) => [el("dt", {}, r), el("dd", {}, v)]));

  const avisos = estado.dados.avisos;
  $("lista-avisos").replaceChildren(...(avisos.length
    ? avisos.map((t) => el("li", {}, t))
    : [el("li", { class: "nenhum" }, "Nenhuma observação.")]));

  const mod = estado.dados.modelo;
  const lim = estado.dados.limiares;
  const linhas = [
    ["Classificador", mod.classificador], ["Segmentação", mod.segmentacao || "não utilizada"],
    ["TorchXRayVision", mod.versao_torchxrayvision], ["Dispositivo", mod.dispositivo],
    ["Limiares", `positivo ≥ ${pct(lim.positivo)} · indeterminado ≥ ${pct(lim.indeterminado)}`],
    ["Tempo de análise", `${estado.dados.tempo_s.toFixed(1)} s`],
  ].filter(([, v]) => v);
  $("dados-modelo").replaceChildren(...linhas.flatMap(([r, v]) => [el("dt", {}, r), el("dd", {}, v)]));
}

// ------------------------------------------------------------------------ //
// Desenho no canvas
// ------------------------------------------------------------------------ //
function redimensionarTela() {
  const dpr = window.devicePixelRatio || 1;
  tela.width = Math.round(tela.clientWidth * dpr);
  tela.height = Math.round(tela.clientHeight * dpr);
}

function ajustarATela() {
  redimensionarTela();
  if (!estado.dados) return;
  const [W, H] = estado.dados.dimensoes_exibicao;
  const largura = tela.clientWidth, altura = tela.clientHeight;
  estado.zoom = Math.min(largura / W, altura / H) * 0.96;
  estado.panX = (largura - W * estado.zoom) / 2;
  estado.panY = (altura - H * estado.zoom) / 2;
  document.querySelectorAll(".cartao").forEach((c) => c.classList.remove("ativo"));
  desenhar();
}

function aplicarTransformacao() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr * estado.zoom, 0, 0, dpr * estado.zoom, dpr * estado.panX, dpr * estado.panY);
}

function caminho(pontos) {
  const p = new Path2D();
  pontos.forEach(([x, y], i) => (i ? p.lineTo(x, y) : p.moveTo(x, y)));
  p.closePath();
  return p;
}

function achadosVisiveis() {
  return estado.dados.achados.filter((a) => estado.visiveis.has(a.chave) && a.status !== "negativo");
}

function desenhar() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, tela.width, tela.height);
  if (!estado.dados || !estado.base) return;
  const [W, H] = estado.dados.dimensoes_exibicao;
  const z = estado.zoom;
  aplicarTransformacao();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.filter = `brightness(${estado.brilho}) contrast(${estado.contraste})${estado.inverter ? " invert(1)" : ""}`;
  ctx.drawImage(estado.base, 0, 0, W, H);
  ctx.filter = "none";

  const visiveis = achadosVisiveis();
  const destaque = estado.destaque;
  const alfaDe = (a) => (destaque && destaque !== a.chave ? 0.25 : 1);

  if (estado.camadas.anatomia) {
    ctx.strokeStyle = "rgba(255,255,255,.55)";
    ctx.lineWidth = 1.2 / z;
    Object.values(estado.dados.contornos_anatomicos).forEach((lista) => lista.forEach((c) => ctx.stroke(caminho(c))));
  }

  if (estado.camadas.mapas) {
    [...visiveis].reverse().forEach((a) => {
      const img = estado.mapas[a.chave];
      if (!img) return;
      ctx.globalAlpha = estado.opacidade * alfaDe(a);
      ctx.drawImage(img, 0, 0, W, H);
    });
    ctx.globalAlpha = 1;
  }

  if (estado.camadas.contornos) {
    visiveis.forEach((a) => {
      ctx.globalAlpha = alfaDe(a);
      ctx.strokeStyle = a.cor;
      ctx.lineWidth = (destaque === a.chave ? 3.5 : 2.2) / z;
      ctx.setLineDash(a.status === "positivo" ? [] : [8 / z, 6 / z]);
      a.regioes.forEach((r) => ctx.stroke(caminho(r.contorno)));
    });
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  const ict = estado.dados.ict;
  if (estado.camadas.ict && ict) {
    const linhas = [[ict.linha_torax, "#4DD0E1"], [ict.linha_coracao, "#EC407A"]];
    ctx.lineWidth = 2 / z;
    linhas.forEach(([[p0, p1], cor]) => {
      ctx.strokeStyle = cor;
      ctx.beginPath();
      ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]);
      [p0, p1].forEach((p) => { ctx.moveTo(p[0], p[1] - 8 / z); ctx.lineTo(p[0], p[1] + 8 / z); });
      ctx.stroke();
    });
    rotulo(`ICT ${ict.indice.toFixed(2).replace(".", ",")}`,
      (ict.linha_coracao[0][0] + ict.linha_coracao[1][0]) / 2, ict.linha_coracao[0][1] + 6 / z, "#EC407A", true);
  }

  if (estado.camadas.rotulos) {
    visiveis.forEach((a) => {
      if (!a.regioes.length) return;
      ctx.globalAlpha = alfaDe(a);
      const [x, y] = a.regioes[0].caixa;
      rotulo(`${a.nome} ${pct(a.escore)}`, x, y - 4 / z, a.cor);
    });
    ctx.globalAlpha = 1;
  }
}

function rotulo(texto, x, y, cor, centralizado = false) {
  const z = estado.zoom;
  const tamanho = 13 / z;
  ctx.font = `600 ${tamanho}px system-ui, sans-serif`;
  const largura = ctx.measureText(texto).width + 10 / z;
  const altura = tamanho + 8 / z;
  const x0 = centralizado ? x - largura / 2 : x;
  const y0 = centralizado ? y : y - altura;
  ctx.fillStyle = cor;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x0, y0, largura, altura, 4 / z); else ctx.rect(x0, y0, largura, altura);
  ctx.fill();
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(cor.slice(i, i + 2), 16));
  ctx.fillStyle = 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#000" : "#fff";
  ctx.textBaseline = "middle";
  ctx.fillText(texto, x0 + 5 / z, y0 + altura / 2);
}

// ------------------------------------------------------------------------ //
// Interação com o canvas
// ------------------------------------------------------------------------ //
let arrasto = null;

tela.addEventListener("wheel", (ev) => {
  if (!estado.dados) return;
  ev.preventDefault();
  const r = tela.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const fator = Math.exp(-ev.deltaY * 0.0015);
  const novo = Math.min(Math.max(estado.zoom * fator, 0.05), 40);
  estado.panX = mx - (mx - estado.panX) * (novo / estado.zoom);
  estado.panY = my - (my - estado.panY) * (novo / estado.zoom);
  estado.zoom = novo;
  desenhar();
}, { passive: false });

tela.addEventListener("pointerdown", (ev) => {
  if (!estado.dados) return;
  arrasto = { x: ev.clientX, y: ev.clientY, panX: estado.panX, panY: estado.panY };
  tela.setPointerCapture(ev.pointerId);
  tela.classList.add("arrastando");
});
tela.addEventListener("pointermove", (ev) => {
  if (arrasto) {
    estado.panX = arrasto.panX + ev.clientX - arrasto.x;
    estado.panY = arrasto.panY + ev.clientY - arrasto.y;
    desenhar();
    return;
  }
  mostrarDica(ev);
});
const soltarArrasto = () => { arrasto = null; tela.classList.remove("arrastando"); };
tela.addEventListener("pointerup", soltarArrasto);
tela.addEventListener("pointercancel", soltarArrasto);
tela.addEventListener("pointerleave", () => { $("dica").hidden = true; });
tela.addEventListener("dblclick", ajustarATela);

function mostrarDica(ev) {
  const dica = $("dica");
  if (!estado.dados) { dica.hidden = true; return; }
  const r = tela.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const px = (ev.clientX - r.left) * dpr, py = (ev.clientY - r.top) * dpr;
  aplicarTransformacao();
  const encontrados = achadosVisiveis().filter((a) => a.regioes.some((reg) => ctx.isPointInPath(caminho(reg.contorno), px, py)));
  if (!encontrados.length) { dica.hidden = true; return; }
  dica.textContent = encontrados.map((a) => `${a.nome} — ${pct(a.escore)} (${STATUS[a.status].toLowerCase()})`).join(" · ");
  dica.style.left = `${ev.clientX - r.left}px`;
  dica.style.top = `${ev.clientY - r.top}px`;
  dica.hidden = false;
}

// ------------------------------------------------------------------------ //
// Controles
// ------------------------------------------------------------------------ //
function selecionarAba(nome) {
  document.querySelectorAll(".abas [role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.aba === nome)));
  const temDados = !!estado.dados;
  $("vazio").hidden = temDados;
  ["achados", "laudo", "escores", "exame"].forEach((n) => { $(`aba-${n}`).hidden = !temDados || n !== nome; });
}

document.querySelectorAll(".abas [role=tab]").forEach((b) => b.addEventListener("click", () => selecionarAba(b.dataset.aba)));

document.querySelectorAll("[data-camada]").forEach((b) => b.addEventListener("click", () => {
  const camada = b.dataset.camada;
  estado.camadas[camada] = !estado.camadas[camada];
  b.setAttribute("aria-pressed", String(estado.camadas[camada]));
  desenhar();
}));

$("btn-inverter").addEventListener("click", (ev) => {
  estado.inverter = !estado.inverter;
  ev.currentTarget.setAttribute("aria-pressed", String(estado.inverter));
  desenhar();
});
$("rng-opacidade").addEventListener("input", (ev) => { estado.opacidade = +ev.target.value; desenhar(); });
$("rng-brilho").addEventListener("input", (ev) => { estado.brilho = +ev.target.value; desenhar(); });
$("rng-contraste").addEventListener("input", (ev) => { estado.contraste = +ev.target.value; desenhar(); });
$("btn-ajustar").addEventListener("click", ajustarATela);

$("btn-abrir").addEventListener("click", () => $("arquivo").click());
$("soltar").addEventListener("click", () => $("arquivo").click());
$("arquivo").addEventListener("change", (ev) => {
  const f = ev.target.files[0];
  if (f) analisar(f);
  ev.target.value = "";
});
$("btn-forcar").addEventListener("click", () => { if (estado.arquivo) analisar(estado.arquivo, true); });
$("btn-fechar-erro").addEventListener("click", esconderErro);

["dragenter", "dragover"].forEach((t) => area.addEventListener(t, (ev) => { ev.preventDefault(); area.classList.add("sobre"); }));
["dragleave", "drop"].forEach((t) => area.addEventListener(t, (ev) => { ev.preventDefault(); area.classList.remove("sobre"); }));
area.addEventListener("drop", (ev) => {
  const f = ev.dataTransfer.files[0];
  if (f) analisar(f);
});

$("btn-pdf").addEventListener("click", () => baixarComEdicao("laudo.pdf", "laudo.pdf"));
$("btn-laudo-dcm").addEventListener("click", () => baixarComEdicao("laudo.dcm", "laudo_pdf.dcm"));
$("btn-copiar").addEventListener("click", copiarTexto);
$("btn-restaurar").addEventListener("click", () => { if (estado.dados) preencherLaudo(estado.dados.laudo); });
$("ed-revisor").addEventListener("input", atualizarSelo);

new ResizeObserver(() => { redimensionarTela(); desenhar(); }).observe(area);

redimensionarTela();
selecionarAba("achados");
