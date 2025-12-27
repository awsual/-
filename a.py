import re

result1 = re.sub(r"\d+", "", "runoob123google456")
result2 = re.sub(r"\d+", "", "run88oob123google456")

print(result1)
print(result2)