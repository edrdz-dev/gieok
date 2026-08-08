"""Parallel English/Spanish corpora for isolating language as a variable.

Each corpus is the same set of facts written twice, with the questions translated too.
Anything that differs between the two runs is therefore attributable to language and not
to content, which is what makes a cross-lingual claim defensible.

``EASY`` holds ten unrelated facts. ``HARD`` holds ten facts from a single domain with
heavily overlapping vocabulary, so telling them apart requires reading the whole sentence
rather than spotting a keyword. Real corpora look like ``HARD``.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParallelCorpus:
    """The same notes and questions in two languages."""

    name: str
    notes_en: tuple[str, ...]
    notes_es: tuple[str, ...]
    queries_en: tuple[tuple[str, int], ...]
    """Question paired with the index of the note that answers it."""
    queries_es: tuple[tuple[str, int], ...]


EASY = ParallelCorpus(
    name="easy",
    notes_en=(
        "The spare key to the flat is kept inside the blue ceramic pot on the balcony.",
        "My bicycle was serviced in March; the mechanic said the rear brake pads need"
        " replacing before winter.",
        "The dentist appointment is on the second Tuesday of every month at half past four"
        " in the afternoon.",
        "Grandmother's paella recipe uses short grain rice, saffron, rabbit and green"
        " beans, never chorizo.",
        "The car insurance policy renews automatically in October unless cancelled thirty"
        " days in advance.",
        "The neighbour's cat is called Mist and is fed twice a day while they are away.",
        "The boiler pressure should sit between one and one point five bar; below that it"
        " needs topping up.",
        "My passport expires in February and renewing it takes roughly six weeks by post.",
        "The garden hose is stored in the shed behind the lawnmower, on a green bracket.",
        "The wifi password was changed last summer and is written under the router.",
    ),
    notes_es=(
        "La llave de repuesto del piso se guarda dentro de la maceta azul del balcon.",
        "Revisaron mi bicicleta en marzo; el mecanico dijo que hay que cambiar las"
        " pastillas del freno trasero antes del invierno.",
        "La cita con el dentista es el segundo martes de cada mes a las cuatro y media"
        " de la tarde.",
        "La receta de paella de la abuela lleva arroz redondo, azafran, conejo y judias"
        " verdes, nunca chorizo.",
        "La poliza del seguro del coche se renueva automaticamente en octubre salvo que"
        " se cancele treinta dias antes.",
        "El gato del vecino se llama Niebla y come dos veces al dia mientras no estan.",
        "La presion de la caldera debe estar entre uno y uno coma cinco bares; por debajo"
        " hay que rellenarla.",
        "Mi pasaporte caduca en febrero y renovarlo tarda unas seis semanas por correo.",
        "La manguera del jardin esta en el cobertizo detras del cortacesped, en un soporte verde.",
        "La contrasena del wifi se cambio el verano pasado y esta apuntada bajo el router.",
    ),
    queries_en=(
        ("Where did I leave the spare key?", 0),
        ("What did the bike mechanic say needed replacing?", 1),
        ("When is my dentist appointment?", 2),
        ("What goes into the family paella?", 3),
        ("When does the car insurance renew?", 4),
        ("What is the name of the cat next door?", 5),
        ("What pressure should the boiler be at?", 6),
        ("How long does a passport renewal take?", 7),
        ("Where is the garden hose kept?", 8),
        ("Where can I find the wifi password?", 9),
    ),
    queries_es=(
        ("¿Donde deje la llave de repuesto?", 0),
        ("¿Que dijo el mecanico de la bici que habia que cambiar?", 1),
        ("¿Cuando es mi cita con el dentista?", 2),
        ("¿Que lleva la paella de la familia?", 3),
        ("¿Cuando se renueva el seguro del coche?", 4),
        ("¿Como se llama el gato del vecino?", 5),
        ("¿A que presion debe estar la caldera?", 6),
        ("¿Cuanto tarda renovar el pasaporte?", 7),
        ("¿Donde se guarda la manguera del jardin?", 8),
        ("¿Donde encuentro la contrasena del wifi?", 9),
    ),
)

HARD = ParallelCorpus(
    name="hard",
    notes_en=(
        "The dentist appointment is the second Tuesday of each month at four thirty in the"
        " afternoon.",
        "The physiotherapy session is the second Thursday of each month at four o'clock in"
        " the afternoon.",
        "The car insurance payment is taken on the second day of each month, around forty euros.",
        "The home insurance payment is taken on the twelfth of each month, around sixty euros.",
        "The boiler service is booked once a year in October and costs about ninety euros.",
        "The chimney sweep comes once a year in November and costs about seventy euros.",
        "The gym membership renews every three months and is charged on a Friday.",
        "The parking permit renews every six months and is charged on a Monday.",
        "The water bill arrives every two months and is usually paid by direct debit.",
        "The electricity bill arrives every month and is usually paid by card.",
    ),
    notes_es=(
        "La cita con el dentista es el segundo martes de cada mes a las cuatro y media"
        " de la tarde.",
        "La sesion de fisioterapia es el segundo jueves de cada mes a las cuatro de la tarde.",
        "El pago del seguro del coche se cobra el dia dos de cada mes, unos cuarenta euros.",
        "El pago del seguro del hogar se cobra el dia doce de cada mes, unos sesenta euros.",
        "La revision de la caldera se reserva una vez al ano en octubre y cuesta unos"
        " noventa euros.",
        "El deshollinador viene una vez al ano en noviembre y cuesta unos setenta euros.",
        "La cuota del gimnasio se renueva cada tres meses y se cobra un viernes.",
        "El permiso de aparcamiento se renueva cada seis meses y se cobra un lunes.",
        "La factura del agua llega cada dos meses y se paga normalmente por domiciliacion.",
        "La factura de la luz llega cada mes y se paga normalmente con tarjeta.",
    ),
    queries_en=(
        ("Which day of the month is the physiotherapy session?", 1),
        ("How much is the home insurance?", 3),
        ("When does the chimney sweep come?", 5),
        ("How often does the parking permit renew?", 7),
        ("How is the electricity bill paid?", 9),
        ("What time is the dentist?", 0),
        ("When is the car insurance charged?", 2),
        ("What does the boiler service cost?", 4),
        ("Which day is the gym charged on?", 6),
        ("How often does the water bill arrive?", 8),
    ),
    queries_es=(
        ("¿Que dia del mes es la sesion de fisioterapia?", 1),
        ("¿Cuanto cuesta el seguro del hogar?", 3),
        ("¿Cuando viene el deshollinador?", 5),
        ("¿Cada cuanto se renueva el permiso de aparcamiento?", 7),
        ("¿Como se paga la factura de la luz?", 9),
        ("¿A que hora es el dentista?", 0),
        ("¿Cuando se cobra el seguro del coche?", 2),
        ("¿Cuanto cuesta la revision de la caldera?", 4),
        ("¿Que dia se cobra el gimnasio?", 6),
        ("¿Cada cuanto llega la factura del agua?", 8),
    ),
)

ALL: tuple[ParallelCorpus, ...] = (EASY, HARD)
