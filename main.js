const $ = (sel) => document.querySelector(sel);
const chatEl = $('#chat');
const inputEl = $('#input');
const providerSel = $('#provider');
const modeSel = $('#mode');
const resetBtn = $('#resetBtn');
const formEl = $('#composer');

let state = {
  provider: 'ollama',
  mode: 'explain',
  messages: [], // {role:'user'|'assistant'|'system', content:string}
};

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function bubble(role, text = '') {
  const wrap = document.createElement('div');
  const isUser = role === 'user';
  wrap.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;
  const b = document.createElement('div');
  b.className = `${isUser ? 'bg-indigo-600 text-white' : 'bg-white border'} max-w-[80%] rounded-2xl px-4 py-3 shadow-sm`;
  b.style.whiteSpace = 'pre-wrap';
  b.textContent = text;
  wrap.appendChild(b);
  chatEl.appendChild(wrap);
  scrollToBottom();
  return b;
}

function addUserMessage(text) {
  state.messages.push({ role: 'user', content: text });
  bubble('user', text);
}

function addAssistantStream() {
  const el = bubble('assistant', '');
  return {
    append(text) {
      el.textContent += text;
      scrollToBottom();
    },
    finalize() {
      state.messages.push({ role: 'assistant', content: el.textContent });
    }
  }
}

async function init() {
  try {
    const res = await fetch('/api/health');
    const info = await res.json();
    state.provider = info.provider;
    state.mode = 'explain';
    providerSel.value = info.provider;
    modeSel.value = state.mode;
  } catch {}
}

providerSel.addEventListener('change', () => {
  state.provider = providerSel.value;
});
modeSel.addEventListener('change', () => {
  state.mode = modeSel.value;
});
resetBtn.addEventListener('click', () => {
  state.messages = [];
  chatEl.innerHTML = '';
});

formEl.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addUserMessage(text);
  const assistant = addAssistantStream();

  // Keep last ~20 turns to limit payload size
  const history = state.messages.slice(-40);

  const payload = {
    provider: state.provider,
    mode: state.mode,
    messages: history,
  };

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resp.ok || !resp.body) {
      assistant.append(`\n[Error] ${resp.status} ${resp.statusText}`);
      assistant.finalize();
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      assistant.append(decoder.decode(value, { stream: true }));
    }
    assistant.finalize();
  } catch (err) {
    assistant.append(`\n[Error] ${err}`);
    assistant.finalize();
  }
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    $('#send').click();
  }
});

init();
