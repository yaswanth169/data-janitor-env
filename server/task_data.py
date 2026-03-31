import random
from typing import Any, Dict, List, Tuple

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zane",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Taylor", "Anderson", "Thomas", "Jackson", "White",
    "Harris", "Martin", "Lee", "Clark", "Lewis", "Walker",
]

DEPARTMENTS = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]

STATES = {
    "California": "CA",
    "Texas": "TX",
    "New York": "NY",
    "Florida": "FL",
    "Illinois": "IL",
    "Pennsylvania": "PA",
    "Ohio": "OH",
    "Georgia": "GA",
    "Michigan": "MI",
    "Washington": "WA",
    "Colorado": "CO",
    "Arizona": "AZ",
}

CITIES_BY_STATE = {
    "CA": ["Los Angeles", "San Francisco", "San Diego", "San Jose"],
    "TX": ["Houston", "Dallas", "Austin", "San Antonio"],
    "NY": ["New York", "Buffalo", "Rochester", "Albany"],
    "FL": ["Miami", "Orlando", "Tampa", "Jacksonville"],
    "IL": ["Chicago", "Springfield", "Naperville", "Peoria"],
    "PA": ["Philadelphia", "Pittsburgh", "Allentown", "Erie"],
    "OH": ["Columbus", "Cleveland", "Cincinnati", "Toledo"],
    "GA": ["Atlanta", "Savannah", "Augusta", "Macon"],
    "MI": ["Detroit", "Grand Rapids", "Ann Arbor", "Lansing"],
    "WA": ["Seattle", "Tacoma", "Spokane", "Bellevue"],
    "CO": ["Denver", "Boulder", "Aurora", "Pueblo"],
    "AZ": ["Phoenix", "Tucson", "Mesa", "Scottsdale"],
}

PRODUCT_NAMES = [
    "Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "Monitor Stand",
    "Webcam HD", "Desk Lamp", "Laptop Stand", "Cable Organizer",
    "Noise-Cancelling Headphones", "Portable Charger", "Smart Speaker",
    "Drawing Tablet", "Ergonomic Chair", "Standing Desk Mat", "Screen Protector",
    "Phone Mount", "Bluetooth Adapter", "External SSD", "Docking Station",
    "Surge Protector", "Desk Fan", "Blue Light Glasses", "Wrist Rest",
    "Mouse Pad XL", "HDMI Cable", "Ethernet Adapter", "Power Strip",
    "Webcam Light", "Document Scanner", "Label Maker",
]

CATEGORIES = ["Electronics", "Accessories", "Furniture", "Peripherals", "Storage"]
SUPPLIERS = ["TechDirect", "OfficePro", "GadgetWorld", "SupplyChain Co", "MegaDistro"]

TaskConfig = Dict[str, Any]
TaskData = Tuple[List[Dict[str, Any]], List[Dict[str, Any]], TaskConfig]


def generate_task1(seed: int = 42) -> TaskData:
    """Employee records cleanup — easy difficulty."""
    rng = random.Random(seed)

    clean_data: List[Dict[str, Any]] = []
    used_ids: set = set()

    for i in range(35):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        while name in used_ids:
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            name = f"{first} {last}"
        used_ids.add(name)

        clean_data.append({
            "employee_id": f"EMP-{i + 1:03d}",
            "name": name,
            "age": rng.randint(22, 62),
            "department": rng.choice(DEPARTMENTS),
            "email": f"{first.lower()}.{last.lower()}@company.com",
            "salary": round(rng.uniform(42000, 145000), 2),
            "hire_date": (
                f"{rng.randint(2016, 2024)}-"
                f"{rng.randint(1, 12):02d}-"
                f"{rng.randint(1, 28):02d}"
            ),
        })

    dirty_data = [dict(row) for row in clean_data]

    dup_indices = rng.sample(range(len(clean_data)), 5)
    for idx in dup_indices:
        dirty_data.append(dict(clean_data[idx]))
    rng.shuffle(dirty_data)

    for row in dirty_data:
        row["age"] = str(row["age"])

    dept_corruptions = {
        "Engineering": ["engineering", "ENGINEERING", "Engg", "engineering"],
        "Marketing": ["marketing", "MARKETING", "Mktg"],
        "Sales": ["sales", "SALES"],
        "HR": ["hr", "Human Resources"],
        "Finance": ["finance", "FINANCE"],
        "Operations": ["operations", "OPERATIONS", "Ops"],
    }
    for row in dirty_data:
        dept = row["department"]
        if dept in dept_corruptions and rng.random() < 0.5:
            row["department"] = rng.choice(dept_corruptions[dept])

    for row in dirty_data:
        if rng.random() < 0.2:
            row["email"] = row["email"].upper()
        if rng.random() < 0.15:
            row["email"] = "  " + row["email"] + "  "

    for row in dirty_data:
        if rng.random() < 0.35:
            row["salary"] = f"${row['salary']:,.2f}"
        elif rng.random() < 0.3:
            row["salary"] = str(row["salary"])

    config: TaskConfig = {
        "task_id": "fix_basics",
        "name": "Fix the Basics",
        "difficulty": "easy",
        "description": (
            "Clean an employee records dataset. Fix duplicate rows, convert "
            "string ages to integers, standardize department names to title case, "
            "clean up email formatting, and convert salary strings to numbers."
        ),
        "max_steps": 15,
        "primary_key": "employee_id",
        "target_schema": {
            "employee_id": "str",
            "name": "str",
            "age": "int",
            "department": "str (title case)",
            "email": "str (lowercase, trimmed)",
            "salary": "float",
            "hire_date": "str (YYYY-MM-DD)",
        },
    }

    return dirty_data, clean_data, config


def generate_task2(seed: int = 123) -> TaskData:
    """Customer contacts normalization — medium difficulty."""
    rng = random.Random(seed)
    state_codes = list(STATES.values())

    clean_data: List[Dict[str, Any]] = []
    for i in range(90):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        state_code = rng.choice(state_codes)
        city = rng.choice(CITIES_BY_STATE[state_code])
        area = rng.choice(["212", "310", "415", "512", "617", "713", "808", "902"])
        phone_raw = f"{area}{rng.randint(2000000, 9999999)}"
        zip_code = f"{rng.randint(10000, 99999)}"
        year = rng.randint(2018, 2024)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        signup = f"{year}-{month:02d}-{day:02d}"

        clean_data.append({
            "id": f"CUST-{i + 1:04d}",
            "first_name": first,
            "last_name": last,
            "email": f"{first.lower()}.{last.lower()}{i}@email.com",
            "phone": f"({phone_raw[:3]}) {phone_raw[3:6]}-{phone_raw[6:]}",
            "city": city,
            "state": state_code,
            "zip_code": zip_code,
            "signup_date": signup,
        })

    dirty_data = [dict(row) for row in clean_data]

    dup_indices = rng.sample(range(len(clean_data)), 10)
    for idx in dup_indices:
        dirty_data.append(dict(clean_data[idx]))
    rng.shuffle(dirty_data)

    date_formats = ["%m/%d/%Y", "%b %d, %Y", "%d-%b-%Y", "%B %d %Y", "%Y-%m-%d"]
    for row in dirty_data:
        fmt = rng.choice(date_formats)
        try:
            dt = datetime_from_iso(row["signup_date"])
            row["signup_date"] = dt.strftime(fmt)
        except Exception:
            pass

    phone_templates = [
        lambda d: f"({d[:3]}) {d[3:6]}-{d[6:]}",
        lambda d: f"{d[:3]}-{d[3:6]}-{d[6:]}",
        lambda d: f"{d[:3]}.{d[3:6]}.{d[6:]}",
        lambda d: d,
        lambda d: f"+1-{d[:3]}-{d[3:6]}-{d[6:]}",
        lambda d: f"1{d}",
    ]
    for row in dirty_data:
        digits = "".join(c for c in row["phone"] if c.isdigit())
        if len(digits) > 10:
            digits = digits[-10:]
        template = rng.choice(phone_templates)
        row["phone"] = template(digits)

    reverse_states = {v: k for k, v in STATES.items()}
    for row in dirty_data:
        if rng.random() < 0.45:
            code = row["state"]
            if code in reverse_states:
                row["state"] = reverse_states[code]

    for row in dirty_data:
        r = rng.random()
        if r < 0.15:
            row["first_name"] = row["first_name"].upper()
        elif r < 0.25:
            row["first_name"] = row["first_name"].lower()
        r = rng.random()
        if r < 0.15:
            row["last_name"] = row["last_name"].upper()
        elif r < 0.25:
            row["last_name"] = row["last_name"].lower()

    for row in dirty_data:
        if rng.random() < 0.2:
            row["email"] = row["email"].upper()
        if rng.random() < 0.1:
            row["email"] = " " + row["email"] + " "

    for row in dirty_data:
        if rng.random() < 0.3:
            row["zip_code"] = int(row["zip_code"])

    config: TaskConfig = {
        "task_id": "normalize_chaos",
        "name": "Normalize the Chaos",
        "difficulty": "medium",
        "description": (
            "Normalize a customer contacts dataset. Remove duplicates, "
            "standardize dates to ISO format, normalize phone numbers to "
            "(XXX) XXX-XXXX format, convert state names to 2-letter codes, "
            "fix name casing to title case, clean emails, and ensure zip codes "
            "are 5-digit strings."
        ),
        "max_steps": 20,
        "primary_key": "id",
        "target_schema": {
            "id": "str",
            "first_name": "str (title case)",
            "last_name": "str (title case)",
            "email": "str (lowercase, trimmed)",
            "phone": "str ((XXX) XXX-XXXX)",
            "city": "str",
            "state": "str (2-letter code)",
            "zip_code": "str (5-digit)",
            "signup_date": "str (YYYY-MM-DD)",
        },
    }

    return dirty_data, clean_data, config


def generate_task3(seed: int = 789) -> TaskData:
    """Sales pipeline merge — hard difficulty.

    Returns (dirty_orders, clean_joined, config).
    The secondary products table is embedded in config["secondary_data"].
    """
    rng = random.Random(seed)

    products: List[Dict[str, Any]] = []
    for i in range(30):
        products.append({
            "product_id": f"PROD-{i + 1:03d}",
            "product_name": PRODUCT_NAMES[i],
            "category": rng.choice(CATEGORIES),
            "supplier": rng.choice(SUPPLIERS),
        })

    clean_joined: List[Dict[str, Any]] = []
    orders_dirty: List[Dict[str, Any]] = []

    cust_names = [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(20)]
    order_num = 0

    for i in range(80):
        prod = rng.choice(products[:25])
        quantity = rng.randint(1, 50)
        unit_price = round(rng.uniform(9.99, 499.99), 2)
        total = round(quantity * unit_price, 2)
        year = rng.randint(2023, 2025)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        order_date = f"{year}-{month:02d}-{day:02d}"
        customer = rng.choice(cust_names)
        order_num += 1

        clean_row = {
            "order_id": f"ORD-{order_num:04d}",
            "product_id": prod["product_id"],
            "customer_name": customer,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": total,
            "order_date": order_date,
            "product_name": prod["product_name"],
            "category": prod["category"],
            "supplier": prod["supplier"],
        }

        dirty_row = {
            "order_id": f"ORD-{order_num:04d}",
            "product_id": prod["product_id"],
            "customer_name": customer,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": total,
            "order_date": order_date,
        }

        orders_dirty.append(dirty_row)
        clean_joined.append(clean_row)

    bad_qty_indices = rng.sample(range(len(orders_dirty)), 8)
    for idx in bad_qty_indices:
        orders_dirty[idx]["quantity"] = rng.choice([-1, 0, -5, 0])

    clean_joined = [
        row for i, row in enumerate(clean_joined) if i not in bad_qty_indices
    ]

    for row in clean_joined:
        row["total"] = round(row["quantity"] * row["unit_price"], 2)

    pid_corruptions = [str.lower, str.lower, lambda s: s.replace("-", ""), str.lower]
    for row in orders_dirty:
        if rng.random() < 0.4:
            fn = rng.choice(pid_corruptions)
            row["product_id"] = fn(row["product_id"])

    for row in orders_dirty:
        if rng.random() < 0.25:
            row["total"] = round(row["total"] * rng.uniform(0.8, 1.3), 2)

    date_formats_pool = ["%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y", "%Y-%m-%d"]
    for row in orders_dirty:
        if rng.random() < 0.5:
            fmt = rng.choice(date_formats_pool)
            try:
                dt = datetime_from_iso(row["order_date"])
                row["order_date"] = dt.strftime(fmt)
            except Exception:
                pass

    for row in orders_dirty:
        if rng.random() < 0.3:
            row["unit_price"] = f"${row['unit_price']:,.2f}"
        elif rng.random() < 0.2:
            row["unit_price"] = str(row["unit_price"])

    name_fns = [str.upper, str.lower, str.title]
    for row in orders_dirty:
        if rng.random() < 0.35:
            row["customer_name"] = rng.choice(name_fns)(row["customer_name"])

    for row in clean_joined:
        row["customer_name"] = row["customer_name"].title()
        row["category"] = row["category"].title()

    config: TaskConfig = {
        "task_id": "pipeline_merge",
        "name": "Pipeline Merge",
        "difficulty": "hard",
        "description": (
            "Merge and reconcile two datasets: orders and products. "
            "Standardize product IDs to uppercase PROD-XXX format, "
            "clean unit prices (remove $ and commas), remove rows with "
            "non-positive quantities, standardize dates, fix customer name "
            "casing, join with the products table on product_id, and "
            "recompute the total column as quantity * unit_price."
        ),
        "max_steps": 30,
        "primary_key": "order_id",
        "secondary_data": products,
        "target_schema": {
            "order_id": "str",
            "product_id": "str (PROD-XXX)",
            "customer_name": "str (title case)",
            "quantity": "int (positive)",
            "unit_price": "float",
            "total": "float (quantity * unit_price)",
            "order_date": "str (YYYY-MM-DD)",
            "product_name": "str",
            "category": "str (title case)",
            "supplier": "str",
        },
    }

    return orders_dirty, clean_joined, config


def datetime_from_iso(iso_str: str):
    from datetime import datetime
    return datetime.strptime(iso_str.strip(), "%Y-%m-%d")


def get_task(task_id: str) -> TaskData:
    tasks = {
        "fix_basics": generate_task1,
        "normalize_chaos": generate_task2,
        "pipeline_merge": generate_task3,
    }
    generator = tasks.get(task_id)
    if not generator:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(tasks.keys())}")
    return generator()


TASK_IDS = ["fix_basics", "normalize_chaos", "pipeline_merge"]
