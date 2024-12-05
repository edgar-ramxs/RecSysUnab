import numpy as np 
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


#  ██╗   ██╗ ██╗
#  ██║   ██║███║
#  ██║   ██║╚██║
#  ╚██╗ ██╔╝ ██║
#   ╚████╔╝  ██║
#    ╚═══╝   ╚═╝


class UserTowerV1(nn.Module):
    def __init__(self, user_input_size: int, embedding_size: int, dropout_rate: float):
        super(UserTowerV1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(user_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        return self.fc(x)


class ItemTowerV1(nn.Module):
    def __init__(self, item_input_size: int, embedding_size: int, dropout_rate: float):
        super(ItemTowerV1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(item_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        return self.fc(x)


#  ██╗   ██╗██████╗ 
#  ██║   ██║╚════██╗
#  ██║   ██║ █████╔╝
#  ╚██╗ ██╔╝██╔═══╝ 
#   ╚████╔╝ ███████╗
#    ╚═══╝  ╚══════╝


class UserTowerV2(nn.Module):
    def __init__(self, user_input_size: int, embedding_size: int, dropout_rate: float):
        super(UserTowerV2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(user_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        return self.fc(x)


class ItemTowerV2(nn.Module):
    def __init__(self, item_input_size: int, embedding_size: int, dropout_rate: float):
        super(ItemTowerV2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(item_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        return self.fc(x)


#  ██╗   ██╗██████╗ 
#  ██║   ██║╚════██╗
#  ██║   ██║ █████╔╝
#  ╚██╗ ██╔╝ ╚═══██╗
#   ╚████╔╝ ██████╔╝
#    ╚═══╝  ╚═════╝ 


class UserTowerV3(nn.Module):
    def __init__(self, user_input_size: int, embedding_size: int, dropout_rate: int):
        super(UserTowerV3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(user_input_size, 256),    # Aumentar la dimensión inicial
            nn.LeakyReLU(negative_slope=0.1),   # Activación LeakyReLU
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),                # Segunda capa con menor dimensión
            nn.LeakyReLU(negative_slope=0.1),   # Activación LeakyReLU
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_size)      # Proyectar al tamaño del embedding
        )

    def forward(self, x):
        return self.fc(x)


class ItemTowerV3(nn.Module):
    def __init__(self, item_input_size: int, embedding_size: int, dropout_rate: float):
        super(ItemTowerV3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(item_input_size, 256),    # Aumentar la dimensión inicial
            nn.LeakyReLU(negative_slope=0.1),   # Activación LeakyReLU
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),                # Segunda capa con menor dimensión
            nn.LeakyReLU(negative_slope=0.1),   # Activación LeakyReLU
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_size)      # Proyectar al tamaño del embedding
        )

    def forward(self, x):
        return self.fc(x)


#  ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗      ██████╗ ███████╗
#  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔═══██╗██╔════╝
#  ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ██║   ██║███████╗
#  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ██║   ██║╚════██║
#  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗╚██████╔╝███████║
#  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚══════╝


class TwoTowerModel(nn.Module):
    def __init__(self, user_input_size: int, item_input_size: int, embedding_size: int, dropout_rate: float, U_Tower=UserTowerV1, I_Tower=ItemTowerV1):
        super(TwoTowerModel, self).__init__()
        self.user_tower = U_Tower(user_input_size, embedding_size, dropout_rate)
        self.item_tower = I_Tower(item_input_size, embedding_size, dropout_rate)

    def forward(self, user_input, item_input):
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        score = torch.sum(user_embedding * item_embedding, dim=1)
        return torch.sigmoid(score)


class TwoTowerModelV1(nn.Module):
    def __init__(self, user_input_size: int, item_input_size: int, embedding_size: int, dropout_rate: float, U_Tower=UserTowerV1, I_Tower=ItemTowerV1):
        super(TwoTowerModelV1, self).__init__()
        self.user_tower = U_Tower(user_input_size, embedding_size, dropout_rate)
        self.item_tower = I_Tower(item_input_size, embedding_size, dropout_rate)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, user_input, item_input):
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        score = self.cosine_similarity(user_embedding, item_embedding)
        return torch.sigmoid(score)


class TwoTowerModelV2(nn.Module):
    def __init__(self, user_input_size: int, item_input_size: int, embedding_size: int, dropout_rate: float, U_Tower=UserTowerV1, I_Tower=ItemTowerV1):
        super(TwoTowerModelV2, self).__init__()
        self.user_tower = U_Tower(user_input_size, embedding_size, dropout_rate)
        self.item_tower = I_Tower(item_input_size, embedding_size, dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, user_input, item_input):
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        combined_embedding = torch.cat((user_embedding, item_embedding), dim=1)
        score = self.fc(combined_embedding).squeeze()
        return torch.sigmoid(score)



#  ██╗███╗   ██╗████████╗████████╗ ██████╗ ██╗    ██╗███████╗██████╗ 
#  ██║████╗  ██║╚══██╔══╝╚══██╔══╝██╔═══██╗██║    ██║██╔════╝██╔══██╗
#  ██║██╔██╗ ██║   ██║      ██║   ██║   ██║██║ █╗ ██║█████╗  ██████╔╝
#  ██║██║╚██╗██║   ██║      ██║   ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
#  ██║██║ ╚████║   ██║      ██║   ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
#  ╚═╝╚═╝  ╚═══╝   ╚═╝      ╚═╝    ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
# https://doi.org/10.1145/3511808.3557072

class IntTower(nn.Module):
    def __init__(self, user_input_size: int, item_input_size: int, embedding_size: int, dropout_rate: float, U_Tower=UserTowerV1, I_Tower=ItemTowerV1) -> None:
        super(IntTower, self).__init__()
        self.user_tower = U_Tower(user_input_size, embedding_size, dropout_rate)
        self.item_tower = I_Tower(item_input_size, embedding_size, dropout_rate)
        self.interaction_layer = nn.Sequential(
            nn.Linear(embedding_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, user_input, item_input):
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        combined_embedding = torch.cat([user_embedding, item_embedding], dim=1)     # Concatenar embeddings
        score = self.interaction_layer(combined_embedding)                          # Score final
        return torch.sigmoid(score).squeeze(1)                                      # Elimina la dimensión extra


#  ██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗     ████████╗██╗    ██╗ ██████╗    ████████╗ ██████╗ ██╗    ██╗███████╗██████╗ 
#  ██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗    ╚══██╔══╝██║    ██║██╔═══██╗   ╚══██╔══╝██╔═══██╗██║    ██║██╔════╝██╔══██╗
#  ██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║       ██║   ██║ █╗ ██║██║   ██║█████╗██║   ██║   ██║██║ █╗ ██║█████╗  ██████╔╝
#  ██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║       ██║   ██║███╗██║██║   ██║╚════╝██║   ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
#  ██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝       ██║   ╚███╔███╔╝╚██████╔╝      ██║   ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
#  ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝        ╚═╝    ╚══╝╚══╝  ╚═════╝       ╚═╝    ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
# https://doi.org/10.1145/3539618.3591643                                                                                                                                   
                                                                                                                                                                                                  
class SparseCrossInteraction(nn.Module):
    def __init__(self, embedding_size: int):
        super(SparseCrossInteraction, self).__init__()
        self.fc = nn.Linear(embedding_size, embedding_size)

    def forward(self, user_embedding, item_embedding):
        cross_interaction = user_embedding * item_embedding
        return self.fc(cross_interaction)

class SparseTwoTower(nn.Module):
    def __init__(self, user_input_size: int, item_input_size: int, embedding_size: int, dropout_rate: float, U_Tower=UserTowerV1, I_Tower=ItemTowerV1):
        super(SparseTwoTower, self).__init__()
        self.user_tower = U_Tower(user_input_size, embedding_size, dropout_rate)
        self.item_tower = I_Tower(item_input_size, embedding_size, dropout_rate)
        self.sparse_cross = SparseCrossInteraction(embedding_size)

    def forward(self, user_input, item_input):
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        sparse_interaction = self.sparse_cross(user_embedding, item_embedding)
        return torch.sigmoid(torch.sum(sparse_interaction, dim=1))


#  ██████╗ ███████╗ ██████╗ ██████╗ ██████╗ 
#  ██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗
#  ██║  ██║█████╗  ██║     ██║   ██║██████╔╝
#  ██║  ██║██╔══╝  ██║     ██║   ██║██╔══██╗
#  ██████╔╝███████╗╚██████╗╚██████╔╝██║  ██║
#  ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝
# https://www.mdpi.com/2076-3417/11/19/8993                                       

class ContextualTower(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(ContextualTower, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        return self.fc(x)

class DECOR(nn.Module):
    def __init__(self, user_input_size: int, item_input_size, context_size: int, embedding_size: int, U_Tower=UserTowerV1, I_Tower=ItemTowerV1):
        super(DECOR, self).__init__()
        self.user_tower = U_Tower(user_input_size, embedding_size)
        self.item_tower = I_Tower(item_input_size, embedding_size)
        self.context_tower = ContextualTower(context_size, embedding_size)
        self.final_layer = nn.Sequential(
            nn.Linear(embedding_size * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_input, item_input, context_input):
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        context_embedding = self.context_tower(context_input)
        combined_embedding = torch.cat([user_embedding, item_embedding, context_embedding], dim=1)
        return torch.sigmoid(self.final_layer(combined_embedding))


