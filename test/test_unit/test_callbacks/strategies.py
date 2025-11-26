"""Composite strategies for callbacks tests."""

from string import ascii_lowercase, ascii_uppercase

from hypothesis import strategies as st

__all__ = [
    "ExampleMetrics",
    "ExampleMetricHistory",
    "example_metric_history",
    "example_metrics",
]

type ExampleMetricHistory = dict[int, float]
type ExampleMetrics = dict[str, ExampleMetricHistory]


@st.composite
def example_metric_history(
    draw: st.DrawFn,
    min_metric_val: float = -20.0,
    max_metric_val: float = 20.0,
    min_num_steps: int = 3,
    max_num_steps: int = 10,
    max_tstep: int = 20,
) -> ExampleMetricHistory:
    """Generate a fake (sparse) metric history."""
    # Generate values
    values = draw(
        st.lists(
            st.floats(min_value=min_metric_val, max_value=max_metric_val),
            min_size=min_num_steps,
            max_size=max_num_steps,
        )
    )

    # Generate fake stepos
    num_samples = len(values)
    steps = draw(
        st.sets(
            st.integers(min_value=0, max_value=max_tstep),
            min_size=num_samples,
            max_size=num_samples,
        )
    )

    # Return dict
    return dict(zip(steps, values))


@st.composite
def example_metrics(
    draw: st.DrawFn,
    min_metric_val: float = -20.0,
    max_metric_val: float = 20.0,
    min_key_size: int = 2,
    max_key_size: int = 10,
    min_num_metrics: int = 3,
    max_num_metrics: int = 10,
    min_num_steps: int = 3,
    max_num_steps: int = 10,
) -> ExampleMetrics:
    """Generate a metrics dictionary."""
    return draw(
        st.dictionaries(
            keys=st.text(
                alphabet=ascii_uppercase + ascii_lowercase,
                min_size=min_key_size,
                max_size=max_key_size,
            ),
            values=example_metric_history(
                min_metric_val=min_metric_val,
                max_metric_val=max_metric_val,
                min_num_steps=min_num_steps,
                max_num_steps=max_num_steps,
            ),
            min_size=min_num_metrics,
            max_size=max_num_metrics,
        )
    )
