

def updatecounter(add, S, u, v, tao, tao_c, tao_u, tao_v):

    neighbourhood_uv = list(S[u] & S[v])

    if add:
        for _ in neighbourhood_uv:
            tao += 1
            tao_c += 1
            tao_u += 1
            tao_v += 1
    else:
        for _ in neighbourhood_uv:
            tao -= 1
            tao_c -= 1
            tao_u -= 1
            tao_v -= 1

    return tao, tao_c, tao_u, tao_v



def estimation(t, M):

    nom = t*(t-1)*(t-2)
    denom = M*(M-1)*(M-2)

    return max(1, nom/denom)


def estimate_global_traingles(t, M, tao, estimate):

    if t < M:
        return tao

    else: 
        return estimate * tao

