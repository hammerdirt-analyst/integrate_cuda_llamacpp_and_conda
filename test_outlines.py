import outlines


model = 'bartkowski/Lama-3.2-3B-Instruct-Q6_K.gguf'


model =  outlines.models.transformers(model, device="cuda")

@outlines.prompt
def customer_support(request):
    """You are an experienced customer success manager.

    Given a request from a client, you need to determine when the
    request is urgent using the label "URGENT" or when it can wait
    a little with the label "STANDARD".

    # Examples

    Request: "How are you?"
    Label: STANDARD

    Request: "I need this fixed immediately!"
    Label: URGENT

    # TASK

    Request: {{ request }}
    Label: """

from outlines.samplers import greedy

generator = outlines.generate.choice(model, ["URGENT", "STANDARD"], sampler=greedy())
requests = [
    "My hair is one fire! Please help me!!!",
    "Just wanted to say hi"
]

prompts = [customer_support(request) for request in requests]

labels = generator(prompts)
print(labels)
# ['URGENT', 'STANDARD']

tokens = generator.stream(prompts)
labels = ["URGENT" if "U" in token else "STANDARD" for token in next(tokens)]
print(labels)
# ['URGENT', 'STANDARD']