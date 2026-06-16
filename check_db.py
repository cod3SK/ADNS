import sqlite3
conn = sqlite3.connect('api/instance/adns_demo.db')
rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
for name, sql in rows:
    print(f"TABLE: {name}")
    print(sql)
    print()
    count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"  rows: {count}")
    print()
conn.close()
