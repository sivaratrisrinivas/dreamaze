const solveButton = document.querySelector("#solve-button");
const result = document.querySelector("#result");
const statusLine = document.querySelector("#status");

const CELL_SIZE = 9;
const FRAME_MS = 750;
const COLORS = {
  start: "#16a34a",
  goal: "#dc2626",
  wall: "#111827",
  open: "#f8fafc",
  solution: "#2563eb",
};

let activeSource = null;
let frameQueue = [];
let playbackTimer = null;
let isPlaying = false;
let currentMaze = null;
let currentStart = null;
let currentGoal = null;
let pendingDoneEvent = null;

function setLoading(isLoading) {
  solveButton.disabled = isLoading;
  solveButton.classList.toggle("is-loading", isLoading);
  statusLine.classList.remove("is-error");
  solveButton.querySelector(".button-label").textContent = isLoading
    ? "Solving..."
    : "Solve New Maze";
  statusLine.textContent = "";
}

function showError(message) {
  stopStream();
  statusLine.classList.add("is-error");
  statusLine.textContent = message;
  setLoading(false);
}

function stopStream() {
  if (activeSource) {
    activeSource.close();
    activeSource = null;
  }
  if (playbackTimer) {
    clearTimeout(playbackTimer);
    playbackTimer = null;
  }
  frameQueue = [];
  isPlaying = false;
  pendingDoneEvent = null;
}

function cellId(row, column) {
  return `maze-cell-${row}-${column}`;
}

function isSameCell(cell, row, column) {
  return cell && cell[0] === row && cell[1] === column;
}

function colorForCell(row, column, mask) {
  if (isSameCell(currentStart, row, column)) return COLORS.start;
  if (isSameCell(currentGoal, row, column)) return COLORS.goal;
  if (mask && mask[row] && mask[row][column]) return COLORS.solution;
  if (currentMaze && currentMaze[row] && currentMaze[row][column]) return COLORS.open;
  return COLORS.wall;
}

function renderMazeShell(event) {
  currentMaze = event.renderedMaze;
  currentStart = event.startCell;
  currentGoal = event.goalCell;
  frameQueue = [];
  isPlaying = false;
  pendingDoneEvent = null;

  const rows = currentMaze.length;
  const columns = rows > 0 ? currentMaze[0].length : 0;
  const width = columns * CELL_SIZE;
  const height = rows * CELL_SIZE;
  const rects = [];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      rects.push(
        `<rect id="${cellId(row, column)}" x="${column * CELL_SIZE}" y="${
          row * CELL_SIZE
        }" width="${CELL_SIZE}" height="${CELL_SIZE}" fill="${colorForCell(
          row,
          column,
          null,
        )}" />`,
      );
    }
  }

  result.hidden = false;
  result.innerHTML = `
    <div class="stream-player">
      <svg
        viewBox="0 0 ${width} ${height}"
        width="${width}"
        height="${height}"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Maze with live Conditional Diffusion Solver trajectory"
      >${rects.join("")}</svg>
      <div class="stream-progress">Step <strong id="stream-step">0</strong> / ${
        event.totalSteps
      }</div>
      <div id="stream-verdict" class="stream-verdict" hidden></div>
    </div>
  `;
}

function applyFrame(event) {
  const mask = event.mask;
  for (let row = 0; row < currentMaze.length; row += 1) {
    for (let column = 0; column < currentMaze[row].length; column += 1) {
      const cell = document.getElementById(cellId(row, column));
      if (cell) {
        cell.setAttribute("fill", colorForCell(row, column, mask));
      }
    }
  }

  const step = document.querySelector("#stream-step");
  if (step) {
    step.textContent = String(event.step);
  }
}

function playQueuedFrames() {
  if (frameQueue.length === 0) {
    isPlaying = false;
    if (pendingDoneEvent) {
      finishSolve(pendingDoneEvent);
      pendingDoneEvent = null;
    }
    return;
  }

  isPlaying = true;
  applyFrame(frameQueue.shift());
  playbackTimer = setTimeout(playQueuedFrames, FRAME_MS);
}

function enqueueFrame(event) {
  frameQueue.push(event);
  if (!isPlaying) {
    playQueuedFrames();
  }
}

function finishSolve(event) {
  const verdict = document.querySelector("#stream-verdict");
  if (verdict) {
    verdict.hidden = false;
    verdict.className = event.validationStatus.includes("Valid")
      ? "stream-verdict valid"
      : "stream-verdict invalid";
    verdict.textContent = event.validationStatus;
  }

  if (event.validationReason) {
    statusLine.textContent = event.validationReason;
  }

  setLoading(false);
  if (activeSource) {
    activeSource.close();
    activeSource = null;
  }
}

function handleStreamEvent(event) {
  if (event.type === "init") {
    renderMazeShell(event);
    return;
  }
  if (event.type === "frame") {
    enqueueFrame(event);
    return;
  }
  if (event.type === "done") {
    if (frameQueue.length > 0 || isPlaying) {
      pendingDoneEvent = event;
    } else {
      finishSolve(event);
    }
    return;
  }
  if (event.type === "error") {
    showError(event.message || "Solve request failed");
  }
}

function solveNewMaze() {
  stopStream();
  setLoading(true);
  result.hidden = true;
  result.innerHTML = "";

  activeSource = new EventSource(`/solve_new_maze_stream?t=${Date.now()}`);
  activeSource.onmessage = (message) => {
    try {
      handleStreamEvent(JSON.parse(message.data));
    } catch (_error) {
      showError("Solve stream returned invalid data");
    }
  };
  activeSource.onerror = () => {
    if (!activeSource) return;
    showError("Solve stream disconnected");
  };
}

solveButton.addEventListener("click", solveNewMaze);
