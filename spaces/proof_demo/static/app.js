const solveButton = document.querySelector("#solve-button");
const result = document.querySelector("#result");
const statusLine = document.querySelector("#status");

function setLoading(isLoading) {
  solveButton.disabled = isLoading;
  solveButton.classList.toggle("is-loading", isLoading);
  statusLine.classList.remove("is-error");
  statusLine.textContent = isLoading
    ? "Running the trained solver on a fresh Grid Maze..."
    : "";
}

function showError(message) {
  statusLine.classList.add("is-error");
  statusLine.textContent = message;
}

function renderResult(html) {
  result.innerHTML = html;

  const scripts = Array.from(result.querySelectorAll("script"));
  for (const script of scripts) {
    const executable = document.createElement("script");
    for (const attribute of script.attributes) {
      executable.setAttribute(attribute.name, attribute.value);
    }
    executable.textContent = script.textContent;
    script.replaceWith(executable);
  }
}

async function solveNewMaze() {
  setLoading(true);

  try {
    const response = await fetch("/solve_new_maze", {
      method: "POST",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`Solve request failed with HTTP ${response.status}`);
    }

    const payload = await response.json();
    if (!payload.html) {
      throw new Error("Solve response did not include visualization HTML");
    }

    renderResult(payload.html);
    statusLine.textContent = "Single-sample solve complete.";
  } catch (error) {
    showError(error instanceof Error ? error.message : "Solve request failed");
  } finally {
    solveButton.disabled = false;
    solveButton.classList.remove("is-loading");
  }
}

solveButton.addEventListener("click", solveNewMaze);
