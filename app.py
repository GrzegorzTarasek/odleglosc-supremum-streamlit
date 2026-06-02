import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.optimize import minimize_scalar


x = sp.Symbol("x")

ALLOWED_LOCALS = {
    "x": x,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}

ALLOWED_FUNCTIONS = {sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.Abs}


def create_function(function_text):
    """Parse text into a SymPy expression and a NumPy-ready function."""
    try:
        expression = sp.sympify(function_text, locals=ALLOWED_LOCALS)
    except (sp.SympifyError, TypeError, SyntaxError) as error:
        raise ValueError(f"Niepoprawny wzór funkcji: {function_text}") from error

    unknown_symbols = expression.free_symbols - {x}
    if unknown_symbols:
        unknown = ", ".join(str(symbol) for symbol in unknown_symbols)
        raise ValueError(f"Wzór zawiera niedozwolone symbole: {unknown}")

    for function_call in expression.atoms(sp.Function):
        if function_call.func not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Wzór zawiera niedozwoloną funkcję: {function_call.func}")

    numeric_function = sp.lambdify(x, expression, modules=["numpy"])
    return expression, numeric_function


def safe_evaluate(function, x_values):
    """Evaluate a function and replace invalid values with NaN."""
    x_array = np.asarray(x_values, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            values = function(x_array)
        except Exception:
            return np.full_like(x_array, np.nan, dtype=float)

    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full_like(x_array, float(values), dtype=float)

    values = np.where(np.isfinite(values), values, np.nan)
    return values


def _scalar_value(function, value):
    values = safe_evaluate(function, np.array([value], dtype=float))
    if values.size == 0 or not np.isfinite(values[0]):
        return np.nan
    return float(values[0])


def _difference_value(f, g, value):
    f_value = _scalar_value(f, value)
    g_value = _scalar_value(g, value)
    if not np.isfinite(f_value) or not np.isfinite(g_value):
        return np.nan
    return abs(f_value - g_value)


def _candidate_indices(h_values, number_of_top_points=30):
    valid_indices = np.flatnonzero(np.isfinite(h_values))
    if valid_indices.size == 0:
        return np.array([], dtype=int)

    local_max_indices = []
    for index in valid_indices:
        left = h_values[index - 1] if index > 0 else -np.inf
        right = h_values[index + 1] if index < len(h_values) - 1 else -np.inf
        current = h_values[index]
        if current >= left and current >= right:
            local_max_indices.append(index)

    top_count = min(number_of_top_points, valid_indices.size)
    top_indices = valid_indices[
        np.argpartition(h_values[valid_indices], -top_count)[-top_count:]
    ]
    endpoint_indices = np.array([0, len(h_values) - 1], dtype=int)

    return np.unique(np.concatenate([local_max_indices, top_indices, endpoint_indices]))


def _optimization_interval(index, x_values):
    left = max(0, index - 1)
    right = min(len(x_values) - 1, index + 1)

    if left == right:
        step = (x_values[-1] - x_values[0]) / max(len(x_values) - 1, 1)
        return (
            max(float(x_values[0]), float(x_values[index] - step)),
            min(float(x_values[-1]), float(x_values[index] + step)),
        )

    return float(x_values[left]), float(x_values[right])


def _deduplicate_points(points, epsilon):
    if not points:
        return []

    rounded_digits = max(3, int(np.ceil(-np.log10(epsilon))) + 2)
    best_by_key = {}
    for point in points:
        key = round(point["x"], rounded_digits)
        if key not in best_by_key or point["h"] > best_by_key[key]["h"]:
            best_by_key[key] = point

    return sorted(best_by_key.values(), key=lambda item: item["x"])


def calculate_supremum_distance(f, g, a, b, epsilon):
    grid_size = 8000
    x_values = np.linspace(a, b, grid_size)
    f_values = safe_evaluate(f, x_values)
    g_values = safe_evaluate(g, x_values)
    h_values = np.abs(f_values - g_values)
    h_values = np.where(np.isfinite(h_values), h_values, np.nan)

    if np.count_nonzero(np.isfinite(h_values)) < 3:
        raise ValueError("Po pominięciu błędnych punktów zostaje zbyt mało danych.")

    candidates = _candidate_indices(h_values)
    optimized_points = []

    for index in candidates:
        if not np.isfinite(h_values[index]):
            continue

        grid_x = float(x_values[index])
        grid_h = float(h_values[index])
        optimized_points.append({"x": grid_x, "h": grid_h})

        lower, upper = _optimization_interval(index, x_values)
        if lower >= upper:
            continue

        def objective(value):
            difference = _difference_value(f, g, value)
            if not np.isfinite(difference):
                return np.inf
            return -difference

        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": max(epsilon / 10, 1e-12)},
        )

        if result.success and np.isfinite(result.fun):
            optimized_points.append({"x": float(result.x), "h": float(-result.fun)})

    if not optimized_points:
        raise ValueError("Nie udało się znaleźć poprawnych kandydatów na maksimum.")

    distance = max(point["h"] for point in optimized_points)
    maximum_points = [
        point for point in optimized_points if point["h"] >= distance - epsilon
    ]
    maximum_points = _deduplicate_points(maximum_points, epsilon)

    max_points_df = pd.DataFrame(
        {
            "x": [point["x"] for point in maximum_points],
            "|f(x)-g(x)|": [point["h"] for point in maximum_points],
        }
    )

    plot_data = {
        "x": x_values,
        "f": f_values,
        "g": g_values,
        "h": h_values,
    }

    return distance, max_points_df, plot_data


def create_functions_plot(plot_data):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=plot_data["x"], y=plot_data["f"], mode="lines", name="f(x)")
    )
    fig.add_trace(
        go.Scatter(x=plot_data["x"], y=plot_data["g"], mode="lines", name="g(x)")
    )
    fig.update_layout(
        title="Wykres funkcji f(x) i g(x)",
        xaxis_title="x",
        yaxis_title="wartość funkcji",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def create_difference_plot(plot_data, max_points_df, distance):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_data["x"],
            y=plot_data["h"],
            mode="lines",
            name="|f(x)-g(x)|",
        )
    )
    fig.add_hline(
        y=distance,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="d∞(f,g)",
        annotation_position="top left",
    )

    if not max_points_df.empty:
        fig.add_trace(
            go.Scatter(
                x=max_points_df["x"],
                y=max_points_df["|f(x)-g(x)|"],
                mode="markers",
                name="punkty maksimum",
                marker=dict(size=10, color="#d62728", symbol="circle"),
            )
        )

    fig.update_layout(
        title="Wykres |f(x)-g(x)| z maksimum",
        xaxis_title="x",
        yaxis_title="|f(x)-g(x)|",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def _render_math_description():
    st.markdown(
        """
        Aplikacja porównuje dwie funkcje na wybranym przedziale `[a, b]`.
        Dla każdego punktu sprawdza, jak bardzo różnią się wartości `f(x)` i
        `g(x)`, a następnie znajduje największą z tych różnic.

        Wynik `d∞(f,g)` oznacza największą wartość `|f(x) - g(x)|` na całym
        badanym przedziale. Na wykresie zaznaczone są punkty, w których ta
        największa różnica jest osiągana z podaną dokładnością.
        """
    )


def main():
    st.set_page_config(
        page_title="Odległość supremum funkcji ciągłych",
        page_icon="📈",
        layout="wide",
    )

    st.title("Odległość supremum funkcji ciągłych")
    _render_math_description()

    with st.sidebar:
        st.header("Dane wejściowe")
        a = st.number_input("a", value=-2.0, format="%.6f")
        b = st.number_input("b", value=2.0, format="%.6f")
        f_text = st.text_input("f(x)", value="sin(x)")
        g_text = st.text_input("g(x)", value="x**2 / 4")
        epsilon = st.number_input("epsilon", value=0.001, min_value=0.0, format="%.8f")
        calculate_clicked = st.button("Oblicz", type="primary")

    if not calculate_clicked:
        st.info("Wpisz dane w panelu bocznym i kliknij Oblicz.")
        return

    if a >= b:
        st.error("Warunek a < b musi być spełniony.")
        return

    if epsilon <= 0:
        st.error("Epsilon musi być dodatni.")
        return

    try:
        f_expression, f = create_function(f_text)
        g_expression, g = create_function(g_text)
        distance, max_points_df, plot_data = calculate_supremum_distance(
            f, g, a, b, epsilon
        )
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Wystąpił nieoczekiwany błąd obliczeń: {error}")
        return

    st.subheader("Wynik")
    st.latex(
        rf"f(x)={sp.latex(f_expression)}, \quad g(x)={sp.latex(g_expression)}"
    )
    st.metric("d∞(f,g)", f"{distance:.10g}")

    st.plotly_chart(create_functions_plot(plot_data), use_container_width=True)
    st.plotly_chart(
        create_difference_plot(plot_data, max_points_df, distance),
        use_container_width=True,
    )

    st.subheader("Punkty maksimum z dokładnością epsilon")
    if max_points_df.empty:
        st.warning("Nie znaleziono punktów spełniających zadany warunek.")
    else:
        st.dataframe(max_points_df, use_container_width=True)


if __name__ == "__main__":
    main()
