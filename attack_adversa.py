import utils.adversa as adversa
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--threshold",
    type=float,
    default=0.99999,
    help="samples with confidence below threshold get attacked until their confidence is higher than the threshold",
)
parser.add_argument(
    "--extension_factor",
    type=float,
    default=0.9,
    help="the size if the cutout from the attacked face",
)
parser.add_argument(
    "--epochs", type=int, default=500, help="number of epochs for the attack"
)
parser.add_argument(
    "--use_alternate_attack",
    type=bool,
    action="store_true",
    default=False,
    help="when True, uses exactly cut out faces instead of square cutouts",
)
args = parser.parse_args()

load_dotenv()
MLSEC_API_KEY = os.getenv("MLSEC_API_KEY")


if args.use_alternate_attack:
    attack_type = adversa.AttackScheduler
else:
    attack_type = adversa.AlternateAttackScheduler
    
    
scheduler = attack_type(
    MLSEC_API_KEY, extension_factor=args.extension_factor
)

threshold = args.threshold


for i in range(100):
    repeaters = [
        (int(el["Name"][0]), int(el["Name"][2]))
        for el in scheduler.db.all()
        if el["confidence"] < threshold
    ]
    print(repeaters)
    if len(repeaters) < 4:
        threshold = 1 - (1 - threshold) / 2.0
        print(f"New threshold is {threshold}")
    for source_id, target_id in repeaters:
        print(f"{source_id}_{target_id}")
        scheduler.attack_pair(source_id, target_id, verbose=False, epochs=args.epochs)
