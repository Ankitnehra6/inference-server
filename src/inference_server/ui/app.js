/* Live view of the scheduler.
 *
 * The one thing this page has to make visible is that the batch is rebuilt on every forward
 * pass. A static batcher and a continuous one look identical in a throughput number and
 * completely different here: static shows a block of columns at one height, then a long thin
 * tail as the batch drains to its slowest member, then a step back up. Continuous stays full.
 *
 * Everything drawn comes from /stats. Nothing is simulated in the browser.
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const el = (id) => document.getElementById(id);

const WIDTH = 1000;
const HEIGHT = 220;
const PADDING = { left: 34, right: 8, top: 10, bottom: 20 };

const POLL_MS = 250;

/* --- server ------------------------------------------------------------------ */

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const body = response.status === 204 ? null : await response.json();
  if (!response.ok) throw new Error(body?.detail ?? response.statusText);
  return body;
}

/* --- timeline ------------------------------------------------------------------ */

/**
 * One column per scheduler step: prefill tokens stacked under decoded sequences.
 *
 * Height is the batch size rather than the token count, because the question the picture
 * answers is "how many sequences were in flight", and a single long prefill would otherwise
 * dwarf every decode column and hide exactly the thing worth seeing.
 */
function drawTimeline(steps, maxSeqs) {
  const svg = el("timeline");
  svg.replaceChildren();
  if (!steps.length) return;

  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const peak = Math.max(maxSeqs, ...steps.map((s) => s.batch + s.waiting));
  const scale = plotHeight / Math.max(1, peak);
  const columnWidth = Math.max(1.5, plotWidth / steps.length);

  steps.forEach((step, index) => {
    const x = PADDING.left + index * columnWidth;
    let y = PADDING.top + plotHeight;

    // Stacked bottom-up: decodes, then prefills, then the queue above the line so backlog is
    // visible without changing the meaning of the batch height.
    for (const [count, className] of [
      [step.decodes, "bar-decode"],
      [step.prefills, "bar-prefill"],
      [Math.min(step.waiting, peak), "bar-waiting"],
    ]) {
      if (!count) continue;
      const height = count * scale;
      y -= height;
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", Math.max(1, columnWidth - 0.5));
      rect.setAttribute("height", height);
      rect.setAttribute("class", className);
      svg.appendChild(rect);
    }
  });

  const axis = document.createElementNS(SVG_NS, "line");
  axis.setAttribute("x1", PADDING.left);
  axis.setAttribute("y1", PADDING.top + plotHeight);
  axis.setAttribute("x2", WIDTH - PADDING.right);
  axis.setAttribute("y2", PADDING.top + plotHeight);
  axis.setAttribute("class", "axis-line");
  svg.appendChild(axis);

  for (const value of [0, Math.round(peak / 2), peak]) {
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", PADDING.left - 6);
    label.setAttribute("y", PADDING.top + plotHeight - value * scale + 3);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "axis-label");
    label.textContent = value;
    svg.appendChild(label);
  }
}

/* --- in-flight requests ---------------------------------------------------------- */

function drawSlots(running, queued) {
  const container = el("slots");
  if (!running.length && !queued.length) {
    container.innerHTML = '<p class="empty">Idle — send some traffic.</p>';
    return;
  }
  container.replaceChildren();

  for (const request of running) {
    const total = request.prompt + request.max_tokens;
    const prefillPct = (request.prefilled / total) * 100;
    const decodePct = (request.generated / total) * 100;

    const row = document.createElement("div");
    row.className = "slot" + (request.preemptions ? " preempted" : "");
    row.innerHTML = `
      <span class="slot-id">${request.id}</span>
      <span class="slot-bar">
        <span class="slot-fill prefill" style="width:${prefillPct}%"></span>
        <span class="slot-fill decode" style="width:${decodePct}%"></span>
      </span>
      <span class="slot-meta">${request.generated}/${request.max_tokens}${
        request.preemptions ? ` ·${request.preemptions}↺` : ""
      }</span>`;
    container.appendChild(row);
  }

  for (const request of queued) {
    const row = document.createElement("div");
    row.className = "slot" + (request.preemptions ? " preempted" : "");
    row.innerHTML = `
      <span class="slot-id">${request.id}</span>
      <span class="slot-bar"></span>
      <span class="slot-meta">queued${request.preemptions ? ` ·${request.preemptions}↺` : ""}</span>`;
    container.appendChild(row);
  }
}

/* --- polling ---------------------------------------------------------------------- */

function render(stats) {
  el("stat-throughput").textContent = stats.throughput_tok_per_s;
  el("stat-running").textContent = stats.counts.running;
  el("stat-queued").textContent = stats.counts.waiting;
  el("stat-cache").textContent = `${Math.round(stats.counts.cache.utilisation * 100)}%`;

  el("stat-ttft50").textContent = `${stats.ttft_p50_ms} ms`;
  el("stat-ttft99").textContent = `${stats.ttft_p99_ms} ms`;
  el("stat-itl50").textContent = `${stats.itl_p50_ms} ms`;
  el("stat-itl99").textContent = `${stats.itl_p99_ms} ms`;

  el("stat-finished").textContent = stats.counts.finished;
  el("stat-steps").textContent = stats.counts.steps.toLocaleString();
  el("stat-preemptions").textContent = stats.counts.preemptions;

  el("stat-model").textContent = stats.model;
  el("stat-scheduler").textContent = stats.scheduler;
  el("config-note").textContent =
    `max ${stats.config.max_num_seqs} sequences, ${stats.config.max_num_batched_tokens} tokens `
    + `per step, chunked prefill ${stats.config.chunked_prefill ? "on" : "off"}. `
    + `KV cache ${stats.counts.cache.capacity_tokens.toLocaleString()} tokens `
    + `in ${stats.counts.cache.num_blocks.toLocaleString()} blocks.`;

  drawTimeline(stats.steps, stats.config.max_num_seqs);
  drawSlots(stats.running, stats.queued);
}

async function poll() {
  try {
    render(await api("/stats"));
  } catch (error) {
    console.warn("stats unavailable", error);
  }
}

/* --- actions ------------------------------------------------------------------------ */

el("load-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  button.disabled = true;
  try {
    const body = {
      count: Number(el("count").value),
      rate: Number(el("rate").value),
      prompt_median: Number(el("prompt").value),
      output_median: Number(el("output").value),
    };
    const result = await api("/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    flash(`Sending ${result.submitted} requests at ${result.rate}/s…`);
  } catch (error) {
    flash(error.message);
  } finally {
    button.disabled = false;
  }
});

function flash(message) {
  const banner = el("banner");
  banner.textContent = message;
  banner.hidden = false;
  clearTimeout(flash.timer);
  flash.timer = setTimeout(() => (banner.hidden = true), 5000);
}

poll();
setInterval(poll, POLL_MS);
