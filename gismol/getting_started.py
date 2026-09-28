from gismol import Attribute, COHObject, GoalConstraint, IdentityConstraint
from gismol.nlp import ChatModule, text_to_coh
from gismol.reasoners import GeneralReasoner, Query

# Build a COH programmatically
coh = COHObject("thermostat")
coh.add_attribute(Attribute("temperature", 20.0, bounds=(0, 60)))
coh.add_identity_constraint(IdentityConstraint("safe", "temperature < 50"))
coh.add_goal_constraint(GoalConstraint("comfort", "temperature", direction="max"))
print(coh.component_report())

# Or compile from natural language
plant = text_to_coh(
    "The core temperature must be less than 900. "
    "If temperature exceeds 800 then increase coolant. "
    "Maximize energy output.",
    name="reactor")

# Reason over it
result = GeneralReasoner().infer(plant, Query("maximize energy output"))
print(result.answer)

# Chat
chat = ChatModule(coh=plant)
print(chat.process("describe the reactor").text)