export default function ({ parentElement, data, setTriggerValue }) {
  const entry = parentElement.querySelector("#question");
  const send = parentElement.querySelector("#send-question");
  const list = parentElement.querySelector("#question-suggestions");
  const suggestions = Array.isArray(data.suggestions) ? data.suggestions : [];
  const disabled = Boolean(data.disabled);
  let activeIndex = -1;
  let matches = [];

  entry.disabled = disabled;
  send.disabled = disabled;

  function closeList() {
    list.hidden = true;
    entry.setAttribute("aria-expanded", "false");
    activeIndex = -1;
  }

  function submit(value) {
    const question = String(value || "").trim();
    if (!question || disabled) return;
    entry.value = "";
    closeList();
    setTriggerValue("question", question);
  }

  function showSuggestions() {
    if (disabled) return closeList();
    const query = entry.value.trim().toLocaleLowerCase();
    matches = suggestions.filter(q => q.toLocaleLowerCase().includes(query));
    list.replaceChildren();
    activeIndex = -1;
    for (const [index, question] of matches.entries()) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "suggestion";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.textContent = question;
      option.onmousedown = e => e.preventDefault();
      option.onclick = () => submit(question);
      list.append(option);
    }
    list.hidden = matches.length === 0;
    entry.setAttribute("aria-expanded", String(matches.length > 0));
  }

  function markActive() {
    [...list.children].forEach((option, index) => {
      option.setAttribute("aria-selected", String(index === activeIndex));
    });
  }

  entry.onfocus = showSuggestions;
  entry.oninput = showSuggestions;
  entry.onkeydown = event => {
    if (event.key === "ArrowDown" && !list.hidden) {
      event.preventDefault();
      activeIndex = (activeIndex + 1) % matches.length;
      markActive();
    } else if (event.key === "ArrowUp" && !list.hidden) {
      event.preventDefault();
      activeIndex = (activeIndex - 1 + matches.length) % matches.length;
      markActive();
    } else if (event.key === "Enter") {
      event.preventDefault();
      submit(activeIndex >= 0 && !list.hidden ? matches[activeIndex] : entry.value);
    } else if (event.key === "Escape") {
      closeList();
    }
  };
  entry.onblur = () => setTimeout(() => {
    if (!parentElement.contains(document.activeElement)) closeList();
  }, 100);
  send.onclick = () => submit(entry.value);

  if (parentElement._lastFocusToken !== data.focus_token) {
    parentElement._lastFocusToken = data.focus_token;
    if (data.focus_token && !disabled) {
      requestAnimationFrame(() => {
        entry.scrollIntoView({ behavior: "smooth", block: "center" });
        entry.focus({ preventScroll: true });
      });
    }
  }
}
