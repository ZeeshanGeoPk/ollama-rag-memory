const state = {
  conversations: [],
  conversationId: null,
  mode: "middleware",
  generating: false,
  controller: null,
  context: null,
};

const elements = {
  conversationList: document.querySelector("#conversation-list"),
  conversationTitle: document.querySelector("#conversation-title"),
  connectionStatus: document.querySelector("#connection-status"),
  messageArea: document.querySelector("#message-area"),
  emptyState: document.querySelector("#empty-state"),
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#message-input"),
  sendButton: document.querySelector("#send-button"),
  modeHint: document.querySelector("#mode-hint"),
  contextMode: document.querySelector("#context-mode"),
  contextTokens: document.querySelector("#context-tokens"),
  contextOriginalTokens: document.querySelector("#context-original-tokens"),
  contextReduction: document.querySelector("#context-reduction"),
  contextChunks: document.querySelector("#context-chunks"),
  contextContent: document.querySelector("#context-content"),
  gpuCompact: document.querySelector("#gpu-compact"),
  gpuCompactText: document.querySelector("#gpu-compact-text"),
  gpuDetails: document.querySelector("#gpu-details"),
  sidebar: document.querySelector("#sidebar"),
  contextPanel: document.querySelector("#context-panel"),
  scrim: document.querySelector("#scrim"),
};

document.querySelector("#new-chat").addEventListener("click", startNewChat);
document.querySelector("#copy-context").addEventListener("click", copyContext);
document.querySelector("#sidebar-toggle").addEventListener("click", () => openPanel("sidebar"));
document.querySelector("#context-toggle").addEventListener("click", () => openPanel("context"));
document.querySelector("#context-close").addEventListener("click", closePanels);
elements.scrim.addEventListener("click", closePanels);

document.querySelectorAll(".mode-option").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.generating) {
    state.controller?.abort();
    return;
  }
  sendMessage();
});

elements.input.addEventListener("input", resizeComposer);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

initialize();

async function initialize() {
  await loadConversations();
  pollGpu();
  window.setInterval(pollGpu, 2000);
  elements.input.focus();
}

async function loadConversations() {
  try {
    state.conversations = await api("/ui/api/conversations");
    renderConversationList();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderConversationList() {
  elements.conversationList.replaceChildren();
  if (!state.conversations.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.style.padding = "10px 8px";
    empty.textContent = "No saved conversations.";
    elements.conversationList.append(empty);
    return;
  }

  state.conversations.forEach((conversation) => {
    const item = document.createElement("div");
    item.className = `conversation-item${conversation.id === state.conversationId ? " active" : ""}`;
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.addEventListener("click", () => openConversation(conversation.id));
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openConversation(conversation.id);
      }
    });

    const copy = document.createElement("span");
    copy.className = "conversation-copy";
    const title = document.createElement("strong");
    title.textContent = conversation.title;
    const preview = document.createElement("span");
    preview.textContent = conversation.preview || "Empty conversation";
    copy.append(title, preview);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "delete-conversation";
    remove.title = "Delete conversation";
    remove.setAttribute("aria-label", "Delete conversation");
    remove.textContent = "×";
    remove.addEventListener("click", async (event) => {
      event.stopPropagation();
      await deleteConversation(conversation.id);
    });

    item.append(copy, remove);
    elements.conversationList.append(item);
  });
}

async function openConversation(conversationId) {
  if (state.generating) return;
  closePanels();
  try {
    const conversation = await api(`/ui/api/conversations/${conversationId}`);
    state.conversationId = conversation.id;
    elements.conversationTitle.textContent = conversation.title;
    elements.messageArea.replaceChildren();
    conversation.messages.forEach((message) => {
      appendMessage(message.role, message.content, message.mode);
    });
    if (!conversation.messages.length) showEmptyState();
    renderConversationList();
    await loadContext(conversationId);
    scrollToBottom();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function startNewChat() {
  if (state.generating) return;
  state.conversationId = null;
  state.context = null;
  elements.conversationTitle.textContent = "New conversation";
  elements.messageArea.replaceChildren();
  showEmptyState();
  renderContext(null);
  renderConversationList();
  closePanels();
  elements.input.focus();
}

async function deleteConversation(conversationId) {
  try {
    await api(`/ui/api/conversations/${conversationId}`, { method: "DELETE" });
    if (state.conversationId === conversationId) startNewChat();
    await loadConversations();
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function sendMessage() {
  const message = elements.input.value.trim();
  if (!message || state.generating) return;

  removeEmptyState();
  appendMessage("user", message, state.mode);
  elements.input.value = "";
  resizeComposer();

  const assistant = appendMessage("assistant", "", state.mode, true);
  setGenerating(true);
  setStatus(state.mode === "middleware" ? "Pruning context..." : "Calling Ollama...");

  state.controller = new AbortController();
  try {
    const response = await fetch("/ui/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: state.conversationId,
        message,
        mode: state.mode,
      }),
      signal: state.controller.signal,
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    await readNdjson(response, (event) => {
      if (event.type === "meta") {
        state.conversationId = event.conversation_id;
        renderContext(event.context);
      } else if (event.type === "token") {
        const shouldFollow = isNearMessageBottom();
        assistant.content.textContent += event.content;
        if (shouldFollow) scrollToBottom();
      } else if (event.type === "done") {
        renderContext(event.context);
        setStatus(formatMetrics(event.metrics));
      } else if (event.type === "error") {
        throw new Error(event.message);
      }
    });
    assistant.root.classList.remove("streaming");
    await loadConversations();
    const active = state.conversations.find((item) => item.id === state.conversationId);
    if (active) elements.conversationTitle.textContent = active.title;
  } catch (error) {
    assistant.root.classList.remove("streaming");
    if (error.name === "AbortError") {
      setStatus("Generation stopped");
      if (!assistant.content.textContent) assistant.root.remove();
    } else {
      assistant.content.textContent = `Request failed: ${error.message}`;
      assistant.root.classList.add("error");
      setStatus("Request failed", true);
    }
  } finally {
    setGenerating(false);
    state.controller = null;
    elements.input.focus();
  }
}

function appendMessage(role, content, mode, streaming = false) {
  removeEmptyState();
  const root = document.createElement("article");
  root.className = `message ${role}${streaming ? " streaming" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "YOU" : "AI";

  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Phi-4 Mini";
  const badge = document.createElement("span");
  badge.textContent = mode === "ollama" ? "Direct Ollama" : "Middle layer";
  label.append(badge);
  const messageContent = document.createElement("div");
  messageContent.className = "message-content";
  messageContent.textContent = content;
  body.append(label, messageContent);
  root.append(avatar, body);
  elements.messageArea.append(root);
  scrollToBottom();
  return { root, content: messageContent };
}

function showEmptyState() {
  const template = document.querySelector("#empty-state");
  if (template && template.parentElement) return;
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.id = "empty-state";
  empty.innerHTML = `
    <div class="empty-mark">LC</div>
    <h1>Start a local conversation</h1>
    <p>Choose whether each reply uses direct Ollama history or the pruned context pipeline.</p>
  `;
  elements.messageArea.append(empty);
}

function removeEmptyState() {
  document.querySelector("#empty-state")?.remove();
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-option").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  elements.modeHint.textContent =
    mode === "middleware" ? "Pruned retrieval enabled" : "Full chat history sent directly";
  elements.contextMode.textContent =
    mode === "middleware" ? "Middle layer" : "Direct Ollama";
}

async function loadContext(conversationId) {
  try {
    const context = await api(`/ui/api/conversations/${conversationId}/context`);
    renderContext(context);
  } catch {
    renderContext(null);
  }
}

function renderContext(context) {
  state.context = context;
  if (!context) {
    elements.contextMode.textContent = state.mode === "middleware" ? "Middle layer" : "Direct Ollama";
    elements.contextTokens.textContent = "0";
    elements.contextOriginalTokens.textContent = "0";
    elements.contextReduction.textContent = "0%";
    elements.contextChunks.textContent = "0";
    elements.contextContent.textContent = "No context has been sent yet.";
    return;
  }
  elements.contextMode.textContent =
    context.mode === "ollama" ? "Direct Ollama" : "Middle layer";
  elements.contextTokens.textContent = formatNumber(context.estimated_tokens || 0);
  elements.contextOriginalTokens.textContent = formatNumber(context.original_tokens || 0);
  elements.contextReduction.textContent = `${context.reduction_percent || 0}%`;
  elements.contextChunks.textContent = formatNumber(context.retrieved_chunks?.length || 0);
  elements.contextContent.textContent =
    context.pruned_context || context.error || "No additional context was needed.";
}

async function copyContext() {
  const text = elements.contextContent.textContent;
  await navigator.clipboard.writeText(text);
  const button = document.querySelector("#copy-context");
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = "Copy"; }, 1200);
}

async function pollGpu() {
  try {
    const stats = await api("/ui/api/gpu");
    renderGpu(stats);
  } catch (error) {
    renderGpu({ available: false, message: error.message, gpus: [] });
  }
}

function renderGpu(stats) {
  elements.gpuCompact.classList.toggle("available", stats.available);
  if (!stats.available || !stats.gpus.length) {
    elements.gpuCompactText.textContent = "GPU unavailable";
    elements.gpuDetails.innerHTML = "";
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = stats.message || "GPU telemetry unavailable.";
    elements.gpuDetails.append(message);
    return;
  }

  const gpu = stats.gpus[0];
  elements.gpuCompactText.textContent = `${Math.round(gpu.utilization_percent || 0)}% GPU`;
  elements.gpuDetails.replaceChildren();

  const name = document.createElement("div");
  name.className = "gpu-name";
  name.textContent = gpu.name;
  elements.gpuDetails.append(name);
  elements.gpuDetails.append(
    meter("Utilization", gpu.utilization_percent || 0, `${Math.round(gpu.utilization_percent || 0)}%`, ""),
    meter(
      "Memory",
      gpu.memory_total_mb ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0,
      `${formatMb(gpu.memory_used_mb)} / ${formatMb(gpu.memory_total_mb)}`,
      "memory",
    ),
  );

  const facts = document.createElement("div");
  facts.className = "gpu-facts";
  const temperature = document.createElement("span");
  temperature.textContent = gpu.temperature_c == null ? "Temp N/A" : `${gpu.temperature_c}°C`;
  const power = document.createElement("span");
  power.textContent = gpu.power_w == null ? "Power N/A" : `${gpu.power_w.toFixed(1)} W`;
  facts.append(temperature, power);
  elements.gpuDetails.append(facts);
}

function meter(label, percent, value, className) {
  const root = document.createElement("div");
  root.className = "gpu-meter";
  const head = document.createElement("div");
  head.className = "gpu-meter-head";
  const name = document.createElement("span");
  name.textContent = label;
  const amount = document.createElement("span");
  amount.textContent = value;
  head.append(name, amount);
  const track = document.createElement("div");
  track.className = "meter-track";
  const fill = document.createElement("div");
  fill.className = `meter-fill ${className}`;
  fill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  track.append(fill);
  root.append(head, track);
  return root;
}

function setGenerating(generating) {
  state.generating = generating;
  elements.sendButton.classList.toggle("stop", generating);
  elements.sendButton.textContent = generating ? "■" : "↑";
  elements.sendButton.title = generating ? "Stop generation" : "Send message";
}

function setStatus(message, isError = false) {
  elements.connectionStatus.textContent = message;
  elements.connectionStatus.style.color = isError ? "var(--danger)" : "";
}

function resizeComposer() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 160)}px`;
}

function scrollToBottom() {
  elements.messageArea.scrollTop = elements.messageArea.scrollHeight;
}

function isNearMessageBottom() {
  const distance =
    elements.messageArea.scrollHeight -
    elements.messageArea.scrollTop -
    elements.messageArea.clientHeight;
  return distance < 80;
}

function openPanel(panel) {
  closePanels();
  elements[panel === "sidebar" ? "sidebar" : "contextPanel"].classList.add("open");
  elements.scrim.classList.add("visible");
}

function closePanels() {
  elements.sidebar.classList.remove("open");
  elements.contextPanel.classList.remove("open");
  elements.scrim.classList.remove("visible");
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function readNdjson(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function formatMetrics(metrics) {
  if (!metrics) return "Complete";
  const duration = metrics.total_duration ? metrics.total_duration / 1_000_000_000 : null;
  const tokens = metrics.eval_count || 0;
  return duration ? `Complete · ${duration.toFixed(1)}s · ${tokens} tokens` : "Complete";
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value);
}

function formatMb(value) {
  if (value == null) return "N/A";
  return value >= 1024 ? `${(value / 1024).toFixed(1)} GB` : `${Math.round(value)} MB`;
}
