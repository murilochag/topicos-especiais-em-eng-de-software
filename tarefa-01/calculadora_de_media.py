def calcular_media(nota1, nota2, nota3):
    media = (nota1 + nota2 + nota3) / 3
    
    if media >= 5.0:
        resultado = "Passou"
    else:
        resultado = "Reprovou"
    
    return f"Média: {media:.2f}\n{resultado}"

print(calcular_media(7.0, 8.0, 9.0))
print(calcular_media(5.0, 7.5, 2.0))