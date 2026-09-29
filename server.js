/**
 * SyntheticFocus — Autonomous multi-persona UX usability testing platform.
 *
 * Backend: Express + Puppeteer (headless Chromium) + Groq (OpenAI-compatible API)
 * Realtime: Server-Sent Events (SSE) streaming screenshots + thought telemetry.
 *
 *   POST /api/start-test    { url, goal, personaIds } -> { testId }
 *   GET  /api/stream/:testId                  SSE: "step" events + terminal "done"
 *   GET  /api/report/:testId                   Executive UX Audit (202 while pending)
 *   GET  /api/analytics                        visitor page-view stats
 *   GET  /api/health                           engine status
 */
require('dotenv').config();

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const express = require('express');
const cors = require('cors');
const puppeteer = require('puppeteer');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3000;

// ---------------------------------------------------------------------------
// Groq client (OpenAI-compatible endpoint). JSON mode via response_format.
// ---------------------------------------------------------------------------
const GROQ_API_KEY = process.env.GROQ_API_KEY;
// Vision model (supports image_url inputs) — qwen/qwen3.8-27b is currently the only
// active Groq model with image input modality (llama-4-scout was deprecated 2026-07-17).
const GROQ_VISION_MODEL = process.env.GROQ_MODEL || 'qwen/qwen3.8-27b';
// Text-only model used for the report synthesis step (no images needed).
const GROQ_TEXT_MODEL = process.env.GROQ_TEXT_MODEL || 'openai/gpt-oss-120b';
const groq = new OpenAI({
  apiKey: GROQ_API_KEY || 'not-set',
  baseURL: 'https://api.groq.com/openai/v1',
});

app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use(trackPageVisit);
app.use(express.static('public'));

// ---------------------------------------------------------------------------
// Visitor analytics (cookie unique IDs + JSON file so restarts keep counts)
// ---------------------------------------------------------------------------
const VISITS_FILE = path.join(__dirname, 'data', 'visits.json');
const MAX_VISIT_EVENTS = 8000;
const VISITOR_COOKIE = 'sf_vid';
let visitEvents = loadVisitEvents();

function loadVisitEvents() {
  try {
    const raw = JSON.parse(fs.readFileSync(VISITS_FILE, 'utf8'));
    return Array.isArray(raw.events) ? raw.events : [];
  } catch {
    return [];
  }
}

function persistVisitEvents() {
  try {
    fs.mkdirSync(path.dirname(VISITS_FILE), { recursive: true });
    fs.writeFileSync(VISITS_FILE, JSON.stringify({ events: visitEvents }));
  } catch (err) {
    console.error('[SyntheticFocus] could not persist visits:', err.message);
  }
}

function parseCookies(req) {
  const out = {};
  for (const part of String(req.headers.cookie || '').split(';')) {
    const i = part.indexOf('=');
    if (i < 0) continue;
    const k = part.slice(0, i).trim();
    const v = part.slice(i + 1).trim();
    if (!k) continue;
    try { out[k] = decodeURIComponent(v); } catch { out[k] = v; }
  }
  return out;
}

function ensureVisitorId(req, res) {
  const existing = parseCookies(req)[VISITOR_COOKIE];
  if (existing && /^[a-zA-Z0-9_-]{8,80}$/.test(existing)) return existing;
  const vid = crypto.randomUUID();
  res.append('Set-Cookie', `${VISITOR_COOKIE}=${encodeURIComponent(vid)}; Path=/; Max-Age=31536000; SameSite=Lax`);
  return vid;
}

function shouldCountVisit(req) {
  if (req.method !== 'GET') return false;
  const p = req.path || '/';
  if (p.startsWith('/api/')) return false;
  if (p !== '/' && !p.endsWith('.html')) return false;
  const ua = String(req.headers['user-agent'] || '');
  if (!ua || /HeadlessChrome|Puppeteer/i.test(ua)) return false;
  return true;
}

function trackPageVisit(req, res, next) {
  try {
    if (shouldCountVisit(req)) {
      const vid = ensureVisitorId(req, res);
      visitEvents.push({ t: Date.now(), path: req.path === '/' ? '/' : req.path, vid });
      if (visitEvents.length > MAX_VISIT_EVENTS) {
        visitEvents = visitEvents.slice(-MAX_VISIT_EVENTS);
      }
      persistVisitEvents();
    }
  } catch { /* never block a page load */ }
  next();
}

function startOfDay(ts) {
  const d = new Date(ts);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function dayKey(ts) {
  const d = new Date(ts);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function uniqueCount(events) {
  return new Set(events.map((e) => e.vid)).size;
}

function summarizeVisits() {
  const now = Date.now();
  const today0 = startOfDay(now);
  const week0 = today0 - 6 * 86400000;
  const today = visitEvents.filter((e) => e.t >= today0);
  const week = visitEvents.filter((e) => e.t >= week0);

  const byPathMap = new Map();
  for (const e of visitEvents) {
    const row = byPathMap.get(e.path) || { path: e.path, views: 0, visitors: new Set() };
    row.views += 1;
    row.visitors.add(e.vid);
    byPathMap.set(e.path, row);
  }
  const byPath = Array.from(byPathMap.values())
    .map((r) => ({ path: r.path, views: r.views, uniqueVisitors: r.visitors.size }))
    .sort((a, b) => b.views - a.views);

  const days = [];
  for (let i = 13; i >= 0; i -= 1) {
    const t = today0 - i * 86400000;
    const key = dayKey(t);
    const slice = visitEvents.filter((e) => dayKey(e.t) === key);
    days.push({ date: key, views: slice.length, uniqueVisitors: uniqueCount(slice) });
  }

  const recent = visitEvents.slice(-40).reverse().map((e) => ({
    t: e.t,
    path: e.path,
    visitor: e.vid.slice(0, 8),
  }));

  return {
    totalViews: visitEvents.length,
    uniqueVisitors: uniqueCount(visitEvents),
    todayViews: today.length,
    todayUnique: uniqueCount(today),
    weekViews: week.length,
    weekUnique: uniqueCount(week),
    byPath,
    days,
    recent,
  };
}

// ---------------------------------------------------------------------------
// Personas
// ---------------------------------------------------------------------------
const PERSONAS = {
  brenda: {
    id: 'brenda',
    name: 'Brenda Miller',
    age: 59,
    trait: 'Low Tech Literacy',
    brief:
      'You are Brenda Miller, 59, a retired school administrator with LOW tech literacy. ' +
      'You do not understand hidden dropdown menus, hamburger menus, or hover-revealed controls. ' +
      'Pop-up modals confuse and frighten you; tiny grey "X" buttons are nearly invisible to you. ' +
      'You abandon tasks quickly when buttons are disabled or labels are unclear. ' +
      'You speak plainly, a little anxiously, and you narrate exactly what confuses you.',
  },
  jax: {
    id: 'jax',
    name: 'Jax Rivera',
    age: 20,
    trait: 'Impatient Digital Native',
    brief:
      'You are Jax Rivera, 20, a chronically-online digital native. You skim pages in seconds and ' +
      'expect pricing and signup to be one tap away. Multi-step forms, gated content, and modals ' +
      'blocking your view make you furious. You rage-quit fast and say so bluntly. ' +
      'Your inner monologue is terse, sarcastic, full of internet slang.',
  },
  marcus: {
    id: 'marcus',
    name: 'Marcus Vance',
    age: 44,
    trait: 'Skeptical Enterprise IT Director',
    brief:
      'You are Marcus Vance, 44, an Enterprise IT Director evaluating vendors. You hunt for ' +
      'data-privacy terms, SOC-2 compliance, refund/cancellation terms, and a clear pricing tier ' +
      'matrix before you trust anything. Vague consent checkboxes and buried legal links are red ' +
      'flags. Your inner monologue is methodical, dry, and risk-focused.',
  },
};

const MAX_STEPS = 8;

// ---------------------------------------------------------------------------
// In-memory test registry + SSE fan-out
// ---------------------------------------------------------------------------
const tests = new Map();   // testId -> { testId, url, goal, personaIds, logs: [], status, report, createdAt }
const streams = new Map(); // testId -> Set<res>

function broadcast(testId, event, payload) {
  const set = streams.get(testId);
  if (!set) return;
  const msg = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
  for (const res of set) {
    try { res.write(msg); } catch { /* client gone */ }
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// DOM Extraction Engine (runs inside the page)
// ---------------------------------------------------------------------------
function extractInteractiveElements() {
  const els = Array.from(
    document.querySelectorAll('a, button, input, select, textarea, [role="button"]')
  );
  const out = [];
  let id = 0;
  for (const el of els) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (rect.width === 0 || rect.height === 0) continue;
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
    el.setAttribute('data-agent-id', String(id));
    const label =
      (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
    out.push({
      id,
      tag: el.tagName.toLowerCase(),
      text: label.slice(0, 80) || null,
      type: el.type || null,
      placeholder: el.placeholder || null,
      href: el.tagName === 'A' && el.getAttribute('href') ? el.getAttribute('href').slice(0, 120) : null,
      disabled: !!el.disabled,
      checked: el.type === 'checkbox' ? !!el.checked : undefined,
      x: Math.round(rect.x), y: Math.round(rect.y),
      w: Math.round(rect.width), h: Math.round(rect.height),
    });
    id += 1;
    if (out.length >= 25) break; // token conservation
  }
  return out;
}

function findLabel(elements, targetId) {
  const el = elements.find((e) => e.id === targetId);
  if (!el) return `#${targetId}`;
  const name = el.text || el.placeholder || el.type || el.tag;
  return `#${targetId}: ${el.tag.toUpperCase()} "${name}"`;
}

// ---------------------------------------------------------------------------
// Groq reasoning call — vision + JSON mode
// ---------------------------------------------------------------------------
async function decideNextStep(persona, goal, pageUrl, elements, history, screenshotB64) {
  const systemPrompt =
    `${persona.brief}\n\n` +
    `You are autonomously usability-testing a website. Your GOAL: "${goal}".\n` +
    `You see the current viewport screenshot plus a numbered list of visible interactive elements ` +
    `(each has a data-agent-id). Decide the SINGLE next action.\n\n` +
    `Rules:\n` +
    `- Prefer the most direct path to the goal. Scroll down only if nothing relevant is visible.\n` +
    `- To type into a field, use action "type" with target_id of the input and input_value.\n` +
    `- Use "complete" ONLY when the goal is verifiably achieved. Use "abandon" when the goal is unreachable or your frustration is maxed.\n` +
    `- internal_thought: 1-2 sentences, raw and uncensored, in your persona's voice. React to what you literally see.\n` +
    `- frustration_level: integer 1 (calm) to 5 (rage-quit). Raise it when blocked, confused, or deceived.\n` +
    `Respond with ONLY a JSON object matching this schema:\n` +
    `{"internal_thought": string, "frustration_level": number, "action": "click"|"type"|"scroll_down"|"complete"|"abandon", ` +
    `"target_id": number|null, "input_value": string|null, "complete_goal": boolean, "abandon_session": boolean, "abandon_reason": string|null}`;

  const historyText = history.length
    ? history.map((h) => `Step ${h.step}: [${h.action}] ${h.target_label || ''} — thought: "${h.thought}" (frustration ${h.frustration}/5)`).join('\n')
    : '(no prior steps)';

  const userText =
    `Current page URL: ${pageUrl}\n\n` +
    `Prior steps this session:\n${historyText}\n\n` +
    `Visible interactive elements (use their data-agent-id as target_id):\n` +
    `${JSON.stringify(elements.map(({ id, tag, text, type, placeholder, disabled, checked }) =>
      ({ id, tag, text, type, placeholder, disabled, checked })), null, 1)}`;

  const completion = await groq.chat.completions.create({
    model: GROQ_VISION_MODEL,
    response_format: { type: 'json_object' },
    temperature: 0.7,
    max_tokens: 600,
    messages: [
      { role: 'system', content: systemPrompt },
      {
        role: 'user',
        content: screenshotB64
          ? [
              { type: 'text', text: userText },
              { type: 'image_url', image_url: { url: `data:image/png;base64,${screenshotB64}` } },
            ]
          : userText,
      },
    ],
  });

  const raw = completion.choices?.[0]?.message?.content || '{}';
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    const m = raw.match(/\{[\s\S]*\}/);
    parsed = m ? JSON.parse(m[0]) : {};
  }
  return {
    internal_thought: String(parsed.internal_thought || '(no thought recorded)'),
    frustration_level: Math.min(5, Math.max(1, parseInt(parsed.frustration_level, 10) || 1)),
    action: ['click', 'type', 'scroll_down', 'complete', 'abandon'].includes(parsed.action) ? parsed.action : 'scroll_down',
    target_id: typeof parsed.target_id === 'number' ? parsed.target_id : null,
    input_value: typeof parsed.input_value === 'string' ? parsed.input_value : null,
    complete_goal: !!parsed.complete_goal,
    abandon_session: !!parsed.abandon_session,
    abandon_reason: parsed.abandon_reason || null,
  };
}

// ---------------------------------------------------------------------------
// Execute one decided action inside the page
// ---------------------------------------------------------------------------
async function executeAction(page, decision) {
  const sel = (id) => `[data-agent-id="${id}"]`;
  try {
    if (decision.action === 'click' && decision.target_id != null) {
      await page.click(sel(decision.target_id));
    } else if (decision.action === 'type' && decision.target_id != null) {
      const s = sel(decision.target_id);
      await page.click(s);
      await page.keyboard.down('Control');
      await page.keyboard.press('KeyA');
      await page.keyboard.up('Control');
      await page.keyboard.press('Backspace');
      await page.type(s, decision.input_value || '');
    } else if (decision.action === 'scroll_down') {
      await page.evaluate(() => window.scrollBy(0, 500));
    }
    // 'complete' / 'abandon' need no page action
  } catch (err) {
    // Non-fatal: element may have vanished after a re-render/navigation.
  }
  await sleep(1400); // let the UI settle before the next observation
}

// ---------------------------------------------------------------------------
// Single persona run
// ---------------------------------------------------------------------------
async function runPersona(browser, test, persona) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });
  const history = [];

  const emit = (entry) => {
    test.logs.push(entry);
    broadcast(test.testId, 'step', entry);
  };

  try {
    await page.goto(test.url, { waitUntil: 'networkidle2', timeout: 30000 });
  } catch {
    // Continue anyway; the agent will observe whatever loaded.
  }
  await sleep(3200); // allow delayed overlays (e.g. newsletter modal) to appear

  for (let step = 1; step <= MAX_STEPS; step += 1) {
    let elements = [];
    try {
      elements = await page.evaluate(extractInteractiveElements);
    } catch {
      elements = [];
    }
    let screenshot = '';
    try {
      screenshot = await page.screenshot({ encoding: 'base64' });
    } catch {
      screenshot = '';
    }

    let decision;
    try {
      decision = await decideNextStep(persona, test.goal, page.url(), elements, history, screenshot);
    } catch (err) {
      emit({
        persona: persona.name, step, thought: `The test engine lost its connection to the reasoning backend (${err.message || 'unknown error'}). I am stopping.`,
        action: 'abandon', frustration: 5, screenshot, status: 'engine_error',
        target_label: null, timestamp: Date.now(),
      });
      break;
    }

    const entry = {
      persona: persona.name,
      step,
      thought: decision.internal_thought,
      action: decision.action,
      frustration: decision.frustration_level,
      screenshot: `data:image/png;base64,${screenshot}`,
      status: 'running',
      target_label: decision.target_id != null ? findLabel(elements, decision.target_id) : null,
      timestamp: Date.now(),
    };
    history.push({ step, action: decision.action, target_label: entry.target_label, thought: entry.thought, frustration: entry.frustration });
    emit(entry);

    const terminal =
      decision.complete_goal || decision.abandon_session || decision.frustration_level >= 5 ||
      decision.action === 'complete' || decision.action === 'abandon';

    if (!terminal) {
      await executeAction(page, decision);
    } else {
      const outcome = decision.complete_goal || decision.action === 'complete' ? 'completed' : 'abandoned';
      emit({
        persona: persona.name, step: step + 0, thought: decision.abandon_reason || entry.thought,
        action: decision.action, frustration: decision.frustration_level,
        screenshot: entry.screenshot, status: outcome,
        target_label: entry.target_label, timestamp: Date.now(),
      });
      break;
    }

    if (step === MAX_STEPS) {
      emit({
        persona: persona.name, step, thought: 'I have run out of steps. The goal remains unreached.',
        action: 'abandon', frustration: entry.frustration,
        screenshot: entry.screenshot, status: 'abandoned',
        target_label: null, timestamp: Date.now(),
      });
    }
  }

  try { await page.close(); } catch { /* noop */ }
}

// ---------------------------------------------------------------------------
// Executive UX Audit synthesis via Groq
// ---------------------------------------------------------------------------
async function synthesizeReport(test) {
  const transcript = test.logs
    .map((l) => `[${l.persona} | step ${l.step} | ${l.action}${l.target_label ? ' ' + l.target_label : ''} | frustration ${l.frustration}/5 | ${l.status}] "${l.thought}"`)
    .join('\n');

  const completion = await groq.chat.completions.create({
    model: GROQ_TEXT_MODEL,
    response_format: { type: 'json_object' },
    temperature: 0.4,
    max_tokens: 2000,
    messages: [
      {
        role: 'system',
        content:
          'You are a senior UX auditor. Given raw session logs from autonomous AI usability-test personas, ' +
          'synthesize an Executive UX Audit. Respond with ONLY a JSON object matching this schema:\n' +
          '{"ux_health_score": number 0-100, "executive_summary": string (one paragraph), ' +
          '"bottlenecks": [{"title": string, "severity": "HIGH"|"MEDIUM"|"LOW", "description": string}], ' +
          '"actionable_fixes": [{"element": string, "issue": string, "fix": string}]}',
      },
      {
        role: 'user',
        content:
          `Testing goal: "${test.goal}"\nTested URL: ${test.url}\n\nSession logs:\n${transcript}\n\n` +
          'Produce the Executive UX Audit JSON.',
      },
    ],
  });

  const raw = completion.choices?.[0]?.message?.content || '{}';
  try {
    return JSON.parse(raw);
  } catch {
    const m = raw.match(/\{[\s\S]*\}/);
    return m ? JSON.parse(m[0]) : { ux_health_score: 0, executive_summary: 'Report synthesis failed.', bottlenecks: [], actionable_fixes: [] };
  }
}

// ---------------------------------------------------------------------------
// Full test orchestration (background)
// ---------------------------------------------------------------------------
async function runTest(test) {
  let browser;
  try {
    browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--single-process'],
    });
  } catch (err) {
    test.status = 'failed';
    test.finishedAt = Date.now();
    broadcast(test.testId, 'done', { status: 'failed', error: `Browser launch failed: ${err.message}` });
    return;
  }

  test.status = 'running';
  try {
    for (const pid of test.personaIds) {
      const persona = PERSONAS[pid];
      if (!persona) continue;
      await runPersona(browser, test, persona);
      await sleep(800);
    }
    test.status = 'synthesizing';
    try {
      test.report = await synthesizeReport(test);
      test.status = 'done';
    } catch (err) {
      test.status = 'done';
      test.report = {
        ux_health_score: 0,
        executive_summary: `Audit synthesis failed: ${err.message}`,
        bottlenecks: [], actionable_fixes: [],
      };
    }
  } finally {
    try { await browser.close(); } catch { /* noop */ }
  }
  broadcast(test.testId, 'done', { status: test.status, reportReady: true });
}

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------
app.get('/api/health', (req, res) => {
  res.json({ ok: true, engine: 'groq', visionModel: GROQ_VISION_MODEL, textModel: GROQ_TEXT_MODEL, groqKeySet: !!GROQ_API_KEY });
});

app.post('/api/start-test', (req, res) => {
  if (!GROQ_API_KEY) {
    return res.status(500).json({ error: 'GROQ_API_KEY is not set. Add it to your .env file.' });
  }
  const { url, goal, personaIds } = req.body || {};
  if (!url || !goal) return res.status(400).json({ error: 'url and goal are required.' });

  const ids = Array.isArray(personaIds) && personaIds.length
    ? personaIds.filter((id) => PERSONAS[id])
    : Object.keys(PERSONAS);
  if (!ids.length) return res.status(400).json({ error: 'No valid personas selected.' });

  const testId = `test_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  const test = { testId, url, goal, personaIds: ids, logs: [], status: 'starting', report: null, createdAt: Date.now() };
  tests.set(testId, test);

  // Fire-and-forget background run
  runTest(test).catch((err) => {
    test.status = 'failed';
    broadcast(testId, 'done', { status: 'failed', error: String(err && err.message || err) });
  });

  res.json({ testId });
});

app.get('/api/stream/:testId', (req, res) => {
  const { testId } = req.params;
  const test = tests.get(testId);
  if (!test) return res.status(404).json({ error: 'Unknown testId.' });

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write(': connected\n\n');

  // Replay history so late joiners see the full session
  for (const log of test.logs) {
    res.write(`event: step\ndata: ${JSON.stringify(log)}\n\n`);
  }

  if (!streams.has(testId)) streams.set(testId, new Set());
  streams.get(testId).add(res);

  if (test.status === 'done' || test.status === 'failed') {
    res.write(`event: done\ndata: ${JSON.stringify({ status: test.status, reportReady: !!test.report })}\n\n`);
  }

  req.on('close', () => {
    const set = streams.get(testId);
    if (set) {
      set.delete(res);
      if (!set.size) streams.delete(testId);
    }
  });
});

app.get('/api/report/:testId', (req, res) => {
  const test = tests.get(req.params.testId);
  if (!test) return res.status(404).json({ error: 'Unknown testId.' });
  if (test.status !== 'done' || !test.report) {
    return res.status(202).json({ status: test.status, message: 'Report is not ready yet.' });
  }
  res.json({ status: 'done', report: test.report, goal: test.goal, url: test.url });
});

app.get('/api/analytics', (_req, res) => {
  res.json(summarizeVisits());
});

app.listen(PORT, () => {
  console.log(`[SyntheticFocus] dashboard   -> http://localhost:${PORT}/`);
  console.log(`[SyntheticFocus] mock target -> http://localhost:${PORT}/mock-site.html`);
  console.log(`[SyntheticFocus] vision model -> ${GROQ_VISION_MODEL} ${GROQ_API_KEY ? '(key present)' : '(NO KEY SET)'}`);
  console.log(`[SyntheticFocus] report model -> ${GROQ_TEXT_MODEL}`);
});
