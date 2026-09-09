/**
 * Ames House Price Predictor — frontend logic.
 *
 * The form is not hard-coded: it is built from GET /schema so the backend
 * stays the single source of truth for field names, defaults and bounds.
 */

(function () {
  "use strict";

  var API_URL = (window.API_URL || "http://localhost:8000").replace(/\/+$/, "");

  var elements = {
    apiStatus: document.getElementById("api-status"),
    fields: document.getElementById("fields"),
    form: document.getElementById("predict-form"),
    submit: document.getElementById("submit-button"),
    reset: document.getElementById("reset-button"),
    result: document.getElementById("result"),
    resultPrice: document.getElementById("result-price"),
    resultMeta: document.getElementById("result-meta"),
    error: document.getElementById("error"),
    errorDetail: document.getElementById("error-detail")
  };

  var schema = null;

  function setStatus(text, modifier) {
    elements.apiStatus.textContent = text;
    elements.apiStatus.className = "status status--" + modifier;
  }

  function showError(message) {
    elements.errorDetail.textContent = message;
    elements.error.hidden = false;
    elements.result.hidden = true;
  }

  function clearError() {
    elements.error.hidden = true;
  }

  function formatPrice(value) {
    return value.toLocaleString("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0
    });
  }

  function buildField(field) {
    var wrapper = document.createElement("div");
    wrapper.className = "field";

    var label = document.createElement("label");
    label.setAttribute("for", field.name);
    label.textContent = field.label;
    wrapper.appendChild(label);

    var input;
    if (field.type === "select") {
      input = document.createElement("select");
      field.options.forEach(function (option) {
        var node = document.createElement("option");
        node.value = option;
        node.textContent = option;
        input.appendChild(node);
      });
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      if (field.minimum !== null) {
        input.min = String(field.minimum);
      }
      if (field.maximum !== null) {
        input.max = String(field.maximum);
      }
    }

    input.id = field.name;
    input.name = field.name;
    input.dataset.fieldType = field.type;
    input.value = field.default;
    input.required = true;
    wrapper.appendChild(input);

    return wrapper;
  }

  function renderForm(payload) {
    schema = payload;
    elements.fields.innerHTML = "";
    payload.fields.forEach(function (field) {
      elements.fields.appendChild(buildField(field));
    });
    elements.submit.disabled = false;
  }

  function resetToDefaults() {
    if (!schema) {
      return;
    }
    schema.fields.forEach(function (field) {
      var input = document.getElementById(field.name);
      if (input) {
        input.value = field.default;
      }
    });
    clearError();
    elements.result.hidden = true;
  }

  function collectPayload() {
    var payload = {};
    schema.fields.forEach(function (field) {
      var input = document.getElementById(field.name);
      payload[field.name] =
        field.type === "number" ? Number(input.value) : input.value;
    });
    return payload;
  }

  function describeFailure(status, body) {
    if (status === 422 && body && Array.isArray(body.detail)) {
      return body.detail
        .map(function (item) {
          var where = (item.loc || []).slice(-1)[0];
          return where + ": " + item.msg;
        })
        .join("; ");
    }
    if (body && typeof body.detail === "string") {
      return body.detail;
    }
    return "HTTP " + status;
  }

  function submit(event) {
    event.preventDefault();
    clearError();
    elements.submit.disabled = true;
    elements.submit.textContent = "Predicting…";

    fetch(API_URL + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload())
    })
      .then(function (response) {
        return response.json().then(function (body) {
          if (!response.ok) {
            throw new Error(describeFailure(response.status, body));
          }
          return body;
        });
      })
      .then(function (body) {
        elements.resultPrice.textContent = formatPrice(body.sale_price);
        elements.resultMeta.textContent =
          "Model output log1p(SalePrice) = " + body.log_price.toFixed(4);
        elements.result.hidden = false;
      })
      .catch(function (err) {
        showError(err.message);
      })
      .finally(function () {
        elements.submit.disabled = false;
        elements.submit.textContent = "Predict price";
      });
  }

  function boot() {
    fetch(API_URL + "/health")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (body) {
        setStatus("API online · v" + body.version, "ok");
        return fetch(API_URL + "/schema");
      })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(renderForm)
      .catch(function (err) {
        setStatus("API unreachable", "down");
        elements.fields.innerHTML = "";
        showError("Cannot reach the backend at " + API_URL + " (" + err.message + ")");
      });
  }

  elements.form.addEventListener("submit", submit);
  elements.reset.addEventListener("click", resetToDefaults);
  boot();
})();
