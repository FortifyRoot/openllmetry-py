from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    achat_wrapper,
    chat_wrapper,
)
from opentelemetry.instrumentation.openai.v1.assistant_async_wrappers import (
    aassistants_create_wrapper,
    amessages_create_wrapper,
    amessages_list_wrapper,
    aruns_create_and_stream_wrapper,
    aruns_create_wrapper,
    aruns_retrieve_wrapper,
)
from opentelemetry.instrumentation.openai.v1.assistant_wrappers import (
    assistants_create_wrapper,
    messages_create_wrapper,
    messages_list_wrapper,
    runs_create_and_stream_wrapper,
    runs_create_wrapper,
    runs_retrieve_wrapper,
)
from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
    realtime_connect_wrapper,
)
from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
    async_responses_cancel_wrapper,
    async_responses_get_or_create_wrapper,
    responses_cancel_wrapper,
    responses_get_or_create_wrapper,
)
from opentelemetry.instrumentation.openai.utils import unwrap_dotted_method


def instrument_beta_safety_wrappers(
    try_wrap,
    tracer,
    *,
    tokens_histogram,
    chat_choice_counter,
    duration_histogram,
    chat_exception_counter,
    streaming_time_to_first_token,
    streaming_time_to_generate,
):
    try_wrap(
        "openai.resources.beta.assistants",
        "Assistants.create",
        assistants_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.assistants",
        "AsyncAssistants.create",
        aassistants_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.chat.completions",
        "Completions.parse",
        chat_wrapper(
            tracer,
            tokens_histogram,
            chat_choice_counter,
            duration_histogram,
            chat_exception_counter,
            streaming_time_to_first_token,
            streaming_time_to_generate,
        ),
    )
    try_wrap(
        "openai.resources.beta.chat.completions",
        "AsyncCompletions.parse",
        achat_wrapper(
            tracer,
            tokens_histogram,
            chat_choice_counter,
            duration_histogram,
            chat_exception_counter,
            streaming_time_to_first_token,
            streaming_time_to_generate,
        ),
    )
    try_wrap(
        "openai.resources.beta.threads.runs",
        "Runs.create",
        runs_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.runs",
        "AsyncRuns.create",
        aruns_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.runs",
        "Runs.retrieve",
        runs_retrieve_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.runs",
        "AsyncRuns.retrieve",
        aruns_retrieve_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.runs",
        "Runs.create_and_stream",
        runs_create_and_stream_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.runs",
        "AsyncRuns.create_and_stream",
        aruns_create_and_stream_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.messages",
        "Messages.create",
        messages_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.messages",
        "AsyncMessages.create",
        amessages_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.messages",
        "Messages.list",
        messages_list_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.threads.messages",
        "AsyncMessages.list",
        amessages_list_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.responses",
        "Responses.create",
        responses_get_or_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.responses",
        "Responses.retrieve",
        responses_get_or_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.responses",
        "Responses.cancel",
        responses_cancel_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.responses",
        "AsyncResponses.create",
        async_responses_get_or_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.responses",
        "AsyncResponses.retrieve",
        async_responses_get_or_create_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.responses",
        "AsyncResponses.cancel",
        async_responses_cancel_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.realtime.realtime",
        "Realtime.connect",
        realtime_connect_wrapper(tracer),
    )
    try_wrap(
        "openai.resources.beta.realtime.realtime",
        "AsyncRealtime.connect",
        realtime_connect_wrapper(tracer),
    )


def uninstrument_beta_safety_wrappers():
    unwrap_dotted_method("openai.resources.beta.assistants", "Assistants.create")
    unwrap_dotted_method("openai.resources.beta.assistants", "AsyncAssistants.create")
    unwrap_dotted_method("openai.resources.beta.chat.completions", "Completions.parse")
    unwrap_dotted_method("openai.resources.beta.chat.completions", "AsyncCompletions.parse")
    unwrap_dotted_method("openai.resources.beta.threads.runs", "Runs.create")
    unwrap_dotted_method("openai.resources.beta.threads.runs", "AsyncRuns.create")
    unwrap_dotted_method("openai.resources.beta.threads.runs", "Runs.retrieve")
    unwrap_dotted_method("openai.resources.beta.threads.runs", "AsyncRuns.retrieve")
    unwrap_dotted_method("openai.resources.beta.threads.runs", "Runs.create_and_stream")
    unwrap_dotted_method("openai.resources.beta.threads.runs", "AsyncRuns.create_and_stream")
    unwrap_dotted_method("openai.resources.beta.threads.messages", "Messages.create")
    unwrap_dotted_method("openai.resources.beta.threads.messages", "AsyncMessages.create")
    unwrap_dotted_method("openai.resources.beta.threads.messages", "Messages.list")
    unwrap_dotted_method("openai.resources.beta.threads.messages", "AsyncMessages.list")
    unwrap_dotted_method("openai.resources.responses", "Responses.create")
    unwrap_dotted_method("openai.resources.responses", "Responses.retrieve")
    unwrap_dotted_method("openai.resources.responses", "Responses.cancel")
    unwrap_dotted_method("openai.resources.responses", "AsyncResponses.create")
    unwrap_dotted_method("openai.resources.responses", "AsyncResponses.retrieve")
    unwrap_dotted_method("openai.resources.responses", "AsyncResponses.cancel")
    unwrap_dotted_method("openai.resources.beta.realtime.realtime", "Realtime.connect")
    unwrap_dotted_method("openai.resources.beta.realtime.realtime", "AsyncRealtime.connect")
