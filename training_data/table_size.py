
import os
TABLE_DIMENSION_MAP = {
        "customer": (45000000, 8),
        "lineitem": (1799989091, 16),
        "nation": (25, 4),
        "region": (5, 3),
        "orders": (450000000, 9),
        "part": (60000000, 9),
        "partsupp": (240000000, 5),
        "supplier": (3000000, 7),
}
CURRENT_FILE_PATH = os.path.dirname(__file__)
TABLE_FILE_PATH = CURRENT_FILE_PATH + "/../TPC-H V3.0.1/dbgen"
for table_name in TABLE_DIMENSION_MAP:
    with open(f"{TABLE_FILE_PATH}/{table_name}.tbl", 'r') as fp:
        line = next(fp)
        num_cols = len(line.split("|"))
        num_lines = sum(1 for line in fp) + 1
        print(table_name, 'Total lines:', num_lines, 'cols', num_cols) # 8
        TABLE_DIMENSION_MAP[table_name] = (num_lines, num_cols)
print(TABLE_DIMENSION_MAP)

'''
customer rows 45000000 cols 8
lineitem rows 1799989091 cols 16
nation rows 25 cols 4
region rows 5 cols 3
orders rows 450000000 cols 9
part rows 60000000 cols 9
partsupp rows 240000000 cols 5
supplier rows 3000000 cols 7



lineitem 1799989091
'''