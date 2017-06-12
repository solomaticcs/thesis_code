import sqlite3

# sqlite
con = sqlite3.connect("wine.sqlite")
cur = con.cursor()

sql = """
SELECT `ID`,`Name` FROM `wine-data-v2-db` WHERE `Name` LIKE ? """
cur.execute(sql, ("%sileni%", ) )

# sql = """
# SELECT `ID`,`Name` FROM `wine-data-v2-db` WHERE `Winery`LIKE ? """
# cur.execute(sql, ("%Hoy%", ))
datas = cur.fetchall()
print type(datas)
for data in datas:
	print data[1]

con.close()