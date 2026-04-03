// ─────────────────────────────────────────────────────────────
//  MBTI Personality Predictor — Backend Server
//  Uses: @huggingface/inference (official npm package)
//  Model: facebook/bart-large-mnli (zero-shot classification)
//
//  SETUP:
//  1. Run:  npm install @huggingface/inference
//  2. Make sure your .env file has:  HF_TOKEN=hf_xxxxxxxx
//  3. Run:  node server.js
// ─────────────────────────────────────────────────────────────

const http = require('http');
const fs   = require('fs');
const path = require('path');

// Load .env
function loadEnv() {
  const envPath = path.join(__dirname, '.env');
  if (!fs.existsSync(envPath)) {
    console.warn('⚠️  No .env file found. Create one with: HF_TOKEN=hf_yourTokenHere');
    return;
  }
  fs.readFileSync(envPath, 'utf-8').split('\n').forEach(line => {
    const [key, ...rest] = line.split('=');
    if (key && rest.length) process.env[key.trim()] = rest.join('=').trim();
  });
}
loadEnv();

const PORT     = 3000;
const HF_TOKEN = process.env.HF_TOKEN || '';

if (!HF_TOKEN) {
  console.error('\n❌  HF_TOKEN missing! Add it to your .env file.\n');
  process.exit(1);
}

// Load HuggingFace inference client
let HfInference;
try {
  ({ HfInference } = require('@huggingface/inference'));
} catch (e) {
  console.error('\n❌  Package not found! Please run:  npm install @huggingface/inference\n');
  process.exit(1);
}

const hf = new HfInference(HF_TOKEN);
console.log('✅  Hugging Face client ready\n');

// ── MBTI type info ────────────────────────────────────────────
const MBTI_INFO = {
  INTJ: { name: "The Architect",    color: "#6366f1", desc: "Strategic, independent, and driven by logic. You see the big picture and have a plan for everything. You value knowledge, competence, and long-term thinking." },
  INTP: { name: "The Thinker",      color: "#8b5cf6", desc: "Analytical, curious, and inventive. You love exploring abstract ideas and theoretical systems. You seek logical explanations for everything around you." },
  ENTJ: { name: "The Commander",    color: "#7c3aed", desc: "Bold, strategic, and a natural leader. You take charge, set ambitious goals, and drive others toward results with confidence and determination." },
  ENTP: { name: "The Debater",      color: "#a855f7", desc: "Quick-witted, inventive, and loves a good debate. You challenge ideas and enjoy intellectual sparring. You thrive on innovation and possibilities." },
  INFJ: { name: "The Advocate",     color: "#c026d3", desc: "Idealistic, empathetic, and deeply insightful. You care deeply about making a positive impact and understanding the complexities of people around you." },
  INFP: { name: "The Mediator",     color: "#db2777", desc: "Imaginative, empathetic, and guided by strong values. You seek meaning and authenticity in everything and have a rich inner world of ideals and creativity." },
  ENFJ: { name: "The Protagonist",  color: "#e11d48", desc: "Charismatic, inspiring, and people-focused. You are a natural mentor who motivates others and is passionate about helping people reach their potential." },
  ENFP: { name: "The Campaigner",   color: "#f59e0b", desc: "Enthusiastic, creative, and free-spirited. You see life as full of possibilities and connect easily with others through warmth and genuine curiosity." },
  ISTJ: { name: "The Logistician",  color: "#0891b2", desc: "Reliable, practical, and detail-oriented. You value traditions, duty, and order. You are dependable and take your responsibilities very seriously." },
  ISFJ: { name: "The Defender",     color: "#0284c7", desc: "Warm, dedicated, and highly observant. You are deeply committed to supporting and protecting those you care about with quiet, steady reliability." },
  ESTJ: { name: "The Executive",    color: "#2563eb", desc: "Organised, logical, and assertive. You excel at managing projects and people, making decisions based on facts and proven methods." },
  ESFJ: { name: "The Consul",       color: "#4f46e5", desc: "Caring, sociable, and eager to help. You are attuned to others' feelings and work hard to create harmony and meet the needs of those around you." },
  ISTP: { name: "The Virtuoso",     color: "#059669", desc: "Observant, rational, and hands-on. You are a master of tools and systems, preferring to learn through direct experience rather than theory." },
  ISFP: { name: "The Adventurer",   color: "#10b981", desc: "Gentle, artistic, and open-minded. You live in the moment, express yourself through creativity, and deeply appreciate beauty in the world around you." },
  ESTP: { name: "The Entrepreneur", color: "#f97316", desc: "Energetic, perceptive, and action-oriented. You love solving problems as they arise and thrive in fast-paced, real-world environments." },
  ESFP: { name: "The Entertainer",  color: "#ef4444", desc: "Spontaneous, energetic, and fun-loving. You are the life of the party, bringing joy to others and embracing new experiences with open arms." },
};

// ── MBTI dimensions ───────────────────────────────────────────
const DIMENSIONS = [
  { key: 'EI', labels: ['extroverted and energized by social interaction', 'introverted and energized by solitude and reflection'], types: ['E', 'I'] },
  { key: 'SN', labels: ['practical and focused on concrete facts and real details', 'intuitive and focused on abstract ideas and future possibilities'], types: ['S', 'N'] },
  { key: 'TF', labels: ['logical and makes decisions based on reason and analysis', 'empathetic and makes decisions based on feelings and values'], types: ['T', 'F'] },
  { key: 'JP', labels: ['organised and prefers structure, planning, and clear decisions', 'spontaneous and prefers flexibility, openness, and adapting as things unfold'], types: ['J', 'P'] },
];

// ── Predict MBTI using official HF package ────────────────────
async function predictMBTI(text) {
  console.log('🤖 Classifying across 4 MBTI dimensions...');

  // Run all 4 dimensions in parallel
  const results = await Promise.all(
    DIMENSIONS.map(dim =>
      hf.zeroShotClassification({
        model: 'facebook/bart-large-mnli',
        inputs: text,
        parameters: { candidate_labels: dim.labels, multi_label: false }
      })
    )
  );

  let mbtiType = '';
  results.forEach((result, i) => {
    const dim    = DIMENSIONS[i];
    const top    = result.labels[0];
    const letter = dim.types[dim.labels.indexOf(top)];
    mbtiType += letter;
    console.log(`  ${dim.key}: ${letter}`);
  });

  const info = MBTI_INFO[mbtiType] || { name: 'Unknown', color: '#8b5cf6', desc: 'Could not determine type.' };
  console.log(`✅  Result: ${mbtiType} — ${info.name}`);
  return { type: mbtiType, name: info.name, color: info.color, description: info.desc };
}

// ── HTTP Server ───────────────────────────────────────────────
const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin',  '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  if (req.method === 'POST' && req.url === '/analyze') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', async () => {
      try {
        const { text } = JSON.parse(body);
        if (!text || !text.trim()) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'No text provided' }));
          return;
        }
        console.log(`\n📩  ${text.trim().split(/\s+/).length} words received`);
        const result = await predictMBTI(text);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (err) {
        console.error('❌', err.message);
        const loading = err.message.toLowerCase().includes('load');
        res.writeHead(loading ? 503 : 500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          error: loading
            ? 'AI model is warming up, please wait 20 seconds and try again.'
            : 'Server error: ' + err.message
        }));
      }
    });
  } else {
    res.writeHead(404); res.end();
  }
});

server.listen(PORT, () => {
  console.log(`🚀  Server running → http://localhost:${PORT}`);
  console.log(`📋  Open index.html in your browser\n`);
});
