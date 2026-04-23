from ollama import chat

response = chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': """
        For pathway IPs transport between nucleus and ER lumen, provide the average values of the molecules in this pathway,
specifically: species_2024018, species_2023929.

You must use ALL information available in the SBML file (unique pathway identifiers,
RDF annotations, compartments, species, parameters, reactions, kinetic laws, rules,
constraints, notes, and metadata) provided below as JSON.

Goal:
Provide a realistic numeric target_value for each species_id.

Output rules (mandatory):
- Reply ONLY with valid JSON, with no extra text.
- Format:
    {
    "pathway": "...",
    "targets": [
        {"species_id": "species_x", "target_value": 0.123},
        ...
    ]
    }
- Include one item for EVERY species_id present.
- target_value must be a real number >= 0.
- Do not invent species_id values that are not in the context.

Full SBML context:
{
  "model_info": {
    "model_id": "pathway_1855192",
    "model_name": "IPs transport between nucleus and ER lumen",
    "model_sbo_term": "",
    "model_notes": "<notes>\n  <p xmlns=\"http://www.w3.org/1999/xhtml\">Inositol phosphate IP6 is imported to the endoplasmic reticulum (ER) lumen from the nucleus (Caffrey et al. 1999). The molecular details of these transport processes remain uncertain.</p>\n</notes>"
  },
  "compartments": [
    {
      "id": "compartment_17957",
      "name": "endoplasmic reticulum lumen",
      "constant": true,
      "units": "",
      "sbo_term": "SBO:0000290",
      "notes": ""
    },
    {
      "id": "compartment_7660",
      "name": "nucleoplasm",
      "constant": true,
      "units": "",
      "sbo_term": "SBO:0000290",
      "notes": ""
    }
  ],
  "species": [
    {
      "species_id": "species_2024018",
      "name": "IP6 [endoplasmic reticulum lumen]",
      "compartment": "compartment_17957",
      "initial_amount": null,
      "initial_concentration": null,
      "boundary_condition": false,
      "has_only_substance_units": false,
      "constant": false,
      "sbo_term": "SBO:0000247",
      "metaid": "metaid_1",
      "notes": "<notes>\n  <p xmlns=\"http://www.w3.org/1999/xhtml\">Derived from a Reactome SimpleEntity. This is a small compound</p>\n</notes>"
    },
    {
      "species_id": "species_2023929",
      "name": "IP6 [nucleoplasm]",
      "compartment": "compartment_7660",
      "initial_amount": null,
      "initial_concentration": null,
      "boundary_condition": false,
      "has_only_substance_units": false,
      "constant": false,
      "sbo_term": "SBO:0000247",
      "metaid": "metaid_3",
      "notes": "<notes>\n  <p xmlns=\"http://www.w3.org/1999/xhtml\">Derived from a Reactome SimpleEntity. This is a small compound</p>\n</notes>"
    }
  ],
  "reactions": [
    {
      "id": "reaction_1855187",
      "name": "IP6 transports from the nucleus to the ER lumen",
      "metaid": "metaid_5",
      "compartment": "compartment_7660",
      "reversible": false,
      "fast": false,
      "sbo_term": "",
      "notes": "<notes>\n  <p xmlns=\"http://www.w3.org/1999/xhtml\">Inositol 1,2,3,4,5,6-hexakisphosphate (IP6) translocates from the nucleus to the endoplasmic reticulum (ER) lumen (Caffrey et al. 1999).</p>\n</notes>",
      "reactants": [
        {
          "species": "species_2023929",
          "stoichiometry": 1.0,
          "constant": true,
          "id": "speciesreference_1855187_input_2023929"
        }
      ],
      "products": [
        {
          "species": "species_2024018",
          "stoichiometry": 1.0,
          "constant": true,
          "id": "speciesreference_1855187_output_2024018"
        }
      ],
      "modifiers": [],
      "kinetic_law": {
        "math": "",
        "formula": "lambda_1 * pow(species_2023929, 1)",
        "local_parameters": []
      }
    },
    {
      "id": "reaction_output_degradation",
      "name": "",
      "metaid": "",
      "compartment": "compartment_17957",
      "reversible": false,
      "fast": false,
      "sbo_term": "",
      "notes": "",
      "reactants": [
        {
          "species": "species_2024018",
          "stoichiometry": 1.0,
          "constant": true,
          "id": ""
        }
      ],
      "products": [],
      "modifiers": [],
      "kinetic_law": {
        "math": "",
        "formula": "K_out * species_2024018",
        "local_parameters": []
      }
    },
    {
      "id": "reaction_input",
      "name": "",
      "metaid": "",
      "compartment": "compartment_7660",
      "reversible": false,
      "fast": false,
      "sbo_term": "",
      "notes": "",
      "reactants": [],
      "products": [
        {
          "species": "species_2023929",
          "stoichiometry": 1.0,
          "constant": true,
          "id": ""
        }
      ],
      "modifiers": [],
      "kinetic_law": {
        "math": "",
        "formula": "K_in",
        "local_parameters": []
      }
    }
  ]
}
    """}],
)
print(response.message.content)