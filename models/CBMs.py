import torch
import torch.nn as nn


class WeightMatrix(nn.Linear):
    def __init__(self, in_features, out_features, activation='sigmoid', act_on_weight=False,
                 init_method='default', concept_bank=None, modality_mask=None, bias=False):
        self.init_method = init_method
        self.concept_bank = concept_bank
        self.modality_mask = modality_mask
        if isinstance(bias, bool):
            super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        else:
            super().__init__(in_features=in_features, out_features=out_features, bias=False)
            self.bias = bias
        self.activation = self.get_act(activation)
        self.arg_activation = self.get_argact(activation)
        self.act_on_weight = act_on_weight

    def __getitem__(self, item):
        return self

    def reset_parameters(self):
        if self.init_method == 'default':
            super().reset_parameters()
        elif self.init_method == 'zero':
            self.weight.data.zero_()
            if self.bias is not None:
                self.bias.data.zero_()
        elif self.init_method == 'one':
            self.weight.data.fill_(1)
            if self.bias is not None:
                self.bias.data.fill_(1)
        elif self.init_method == 'kaiming':
            torch.nn.init.kaiming_normal_(self.weight)
        elif self.init_method == 'concept' and self.concept_bank is not None:
            self.weight.data.copy_(self.concept_bank.cls_weight)
            # self.init_weight.scatter_(0, concept2cls, self.cfg.init_val)
        else:
            raise NotImplementedError(self.init_method)

    def get_act(self, activation):
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'softmax':
            return nn.Softmax(dim=-1)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation is None:
            return lambda x: x
        elif activation == 'None':
            return lambda x: x
        else:
            raise NotImplementedError(activation)

    def get_argact(self, activation):
        if activation == 'sigmoid':
            return lambda x: torch.log(x / (1 - x))
        elif activation is None:
            return lambda x: x
        elif activation == 'None':
            return lambda x: x
        else:
            raise NotImplementedError(activation)

    def attention_matrix(self, input):
        # input (B, 1, H)
        q = input.unsqueeze(1)
        # weight (1, class, H)
        if self.act_on_weight:
            # attention matrix (B, class, H)
            attention_matrix = q * self.activation(self.weight).unsqueeze(0)
        else:
            k = self.weight.unsqueeze(0)
            # attention score (B, class, H)
            attention_matrix = self.activation(q * k)
        return attention_matrix

    def concept_score_from_attn(self, attention_score, cls=None):
        # attention_score (B, H)
        if attention_score.ndim == 3:
            cls = attention_score.sum(-1).argmax()
            attention_score = attention_score[:, cls]
        elif cls is None:
            raise NotImplementedError
        if self.act_on_weight:
            # attention score (B, class, H)
            weight = self.activation(self.weight)[cls].unsqueeze(0)
        else:
            attention_score = self.arg_activation(attention_score)
            # weight (1, H)
            weight = self.weight[cls].unsqueeze(0)
        # concept_score (B, H)
        concept_score = attention_score / weight
        return concept_score

    def classification(self, attention_matrix):
        # 直接用 Attention Score 求 sum 分类
        return attention_matrix.sum(-1) if self.bias is None else attention_matrix.sum(-1) + self.bias

    def forward(self, inp):
        attention_matrix = self.attention_matrix(inp)
        return self.classification(attention_matrix)


class MMWeightMatrix(nn.Module):
    def __init__(self, modality_feature_dict,
                 modality_map=None,
                 activation='sigmoid',
                 act_on_weight=False,
                 init_method='zero',
                 bias=False):
        super().__init__()
        if modality_map is None:
            modality_map = {'FA': 'FA', 'ICGA': 'ICGA', 'US': 'US'}
        self.modality_map = modality_map
        self.mm_order = list(dict.fromkeys(modality_map.values()))
        if bias:
            self.bias = nn.Parameter(torch.empty(3))
        else:
            self.bias = False
        self.modality_weight_dict = nn.ModuleDict(
            {m: WeightMatrix(in_features=modality_feature_dict[m][0], out_features=modality_feature_dict[m][1],
                             activation=activation, act_on_weight=act_on_weight,
                             init_method=init_method, bias=self.bias) for m in self.modality_map.values()}
        )

    @property
    def weight(self):
        return torch.concat([self.modality_weight_dict[m].weight for m in self.modality_map.values()], dim=-1)

    def __getitem__(self, modality):
        if 'MM' in modality:
            return self
        self.modality = self.modality_map[modality]
        return self.modality_weight_dict[self.modality]

    def inp2modality(self, inp):
        chunks = torch.split(inp,
                             split_size_or_sections=[
                                 self.modality_weight_dict[m].weight.shape[1] for m in self.mm_order
                             ], dim=-1)
        return chunks

    def attention_matrix(self, inp):
        attention_matrix = []
        for modality, x in zip(self.mm_order, self.inp2modality(inp)):
            attention_matrix.append(self.modality_weight_dict[modality].attention_matrix(x))
        attention_matrix = torch.concat(attention_matrix, dim=-1)
        return attention_matrix

    def concept_score_from_attn(self, attention_score, cls=None):
        concept_score = []
        for modality, x in zip(self.mm_order, self.inp2modality(attention_score)):
            concept_score.append(self.modality_weight_dict[modality].concept_score_from_attn(x, cls))
        concept_score = torch.concat(concept_score, dim=-1)
        return concept_score

    def classification(self, attention_matrix):
        # 直接用 Attention Score 求 sum 分类
        return attention_matrix.sum(-1) if self.bias is False else attention_matrix.sum(-1) + self.bias

    def forward(self, inp):
        attention_matrix = self.attention_matrix(inp)
        return self.classification(attention_matrix)


class CBM(nn.Module):
    def __init__(self, concept_bank, idx_to_class=None, n_classes=5, fusion='max',
                 analysis_top_k=5, modality_mask=False,
                 analysis_threshold=0):
        super().__init__()
        self.fusion = fusion
        self.concept_bank = concept_bank
        self.mm_order = ['FA', 'ICGA', 'US']
        self.hooks = []
        self.n_classes = n_classes
        self.modality_mask = modality_mask
        self.top_k = analysis_top_k
        self.threshold = analysis_threshold
        # Will be used to plot classifier weights nicely
        self.idx_to_class = idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}
        self.mode = None
        print(f'fusion: {self.fusion} hidden_dim: {concept_bank.hidden_dim} n_classes: {self.n_classes}')

    @property
    def weight(self):
        return self.classifier.weight

    def register_hook(self, fn):
        self.hooks.append(fn)

    def remove_hook(self, fn=None):
        if fn is None:
            self.hooks = []
        else:
            self.hooks.remove(fn)

    def run_hooks(self, x):
        for fn in self.hooks:
            x = fn(x)

    def pooling_fusion(self, x):
        # x (B, T, H)
        if self.fusion == 'max':
            return x.max(dim=1)[0]
        elif self.fusion == 'mean':
            return x.mean(dim=1)
        elif self.fusion == 'sum':
            return x.sum(dim=1)
        else:
            raise NotImplementedError(self.fusion)

    def concept_fusion(self, x: dict):
        # x (B, T, H)
        x = {k: self.pooling_fusion(v) for k, v in x.items()}
        self.run_hooks(x)
        mode = self.concept_bank.__name__ if self.mode is None else self.mode
        if mode == 'concat' or 'Multi' in mode:
            x = torch.concat([x[k] for k in self.mm_order if k in x.keys()], dim=-1)
        elif mode == 'dict':
            x = {k: x[k] for k in self.mm_order if k in x.keys()}
        else:
            # modality pooling
            x = torch.stack([x[k] for k in self.mm_order if k in x.keys()], dim=1)
            x = self.pooling_fusion(x)
        return x

    def concept_score(self, inp):
        out = {}
        m_mask = self.modality_mask and len(inp) > 1
        for modality, x in inp.items():
            B = x.shape[0]
            x = x.reshape(-1, *x.shape[2:])
            x = self.concept_bank[modality].compute_dist(x, m_mask=m_mask)
            x = x.reshape(B, -1, x.shape[-1])
            out[modality] = x
        return out

    def analyze_classifier(self, modality, cls, attention_matrix,
                           names=None,
                           pathologies=None,
                           imgs=None,
                           print_lows=False):
        pred = torch.softmax(cls.classification(attention_matrix), dim=1).argmax(-1).cpu().numpy()
        output = []
        sign = 1
        if print_lows:
            sign = -1
        for b in range(pred.shape[0]):
            idx = pred[b]
            img = imgs[b] if imgs is not None else ''
            cls_name = self.idx_to_class[idx]
            attn_score = sign * attention_matrix[b][idx]

            if isinstance(self.top_k, int):
                scores, concepts = self.concept_bank[modality].get_topk_concepts(attn_score, k=self.top_k,
                                                                                 sign=sign)
            else:
                scores, concepts = self.concept_bank[modality].get_concept_from_threshold(
                    attn_score,
                    self.threshold
                )

            for j, concept in enumerate(concepts):
                output.append(
                    {"id": j, "name": names[b] if names else '',
                     "pathology": pathologies[b] if pathologies else '',
                     "modality": modality,
                     "img": img if img else '',
                     "pred": cls_name,
                     "concept": concept,
                     "score": f"{scores[j]:.3f}"}
                )
        return output

    def infer(self, inp, modality, f_names, imgs=None):
        if 'MM' not in modality:
            inp = {modality: inp[modality]}
        analysis = []
        pathologies, names = list(zip(*map(lambda x: x.split('_'), f_names)))
        inp = self.concept_score(inp)
        if imgs is not None:
            # image level
            for m, sc in inp.items():
                for t, img in enumerate(imgs[m]):
                    attention_matrix = self.classifier[m].attention_matrix(sc[:, t, :])
                    analysis.extend(self.analyze_classifier(names=names,
                                                            pathologies=pathologies,
                                                            modality=m,
                                                            imgs=img,
                                                            cls=self.classifier[m],
                                                            attention_matrix=attention_matrix,
                                                            print_lows=False))

        # multi level
        score = self.concept_fusion(inp)
        cls = self.classifier[modality]
        attention_matrix = cls.attention_matrix(score)
        analysis.extend(self.analyze_classifier(names=names,
                                                pathologies=pathologies,
                                                modality=modality,
                                                imgs=None,
                                                cls=cls,
                                                attention_matrix=attention_matrix,
                                                print_lows=False))
        return cls.classification(attention_matrix), analysis

    def attention_matrix(self, inp, modality):
        inp: dict = self.concept_score(inp)
        inp = self.concept_fusion(inp)
        return self.classifier[modality].attention_matrix(inp)

    def attention_matrix_from_modified_attention_score(self, attention_score, modality, cls=None):
        concept_score = self.classifier[modality].concept_score_from_attn(attention_score, cls)
        attention_matrix = self.classifier[modality].attention_matrix(concept_score)
        return attention_matrix

    def predict_from_modified_attention_score(self, attention_score, modality, cls=None):
        attention_matrix = self.attention_matrix_from_modified_attention_score(attention_score, modality, cls)
        return self.classification(attention_matrix, modality)

    def classification(self, attention_matrix, modality):
        return self.classifier[modality].classification(attention_matrix)

    def forward_mm(self, inp, modality):
        pass

    def forward(self, inp, modality):
        if 'MM' not in modality:
            inp = {modality: inp[modality]}
        return self.forward_mm(inp, modality)


class SLinearCBM(CBM):
    def __init__(self, concept_bank, idx_to_class=None, n_classes=5, fusion='max', activation='sigmoid',
                 analysis_top_k=5,
                 analysis_threshold=0,
                 act_on_weight=False,
                 init_method='default',
                 modality_mask=False,
                 bias=False,
                 **kwargs
                 ):
        """
        PosthocCBM Linear Layer.
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
            mode (str, optional): Mode of the MM classifier. Defaults to 'c'. ( 'r':residual, 'b': clack)
        """
        super(SLinearCBM, self).__init__(concept_bank,
                                         idx_to_class=idx_to_class,
                                         n_classes=n_classes,
                                         fusion=fusion,
                                         analysis_top_k=analysis_top_k,
                                         analysis_threshold=analysis_threshold,
                                         modality_mask=modality_mask,
                                         **kwargs
                                         )
        # A single linear layer will be used as the classifier
        self.classifier = WeightMatrix(self.concept_bank['MM'].n_concepts,
                                       self.n_classes,
                                       activation=activation,
                                       act_on_weight=act_on_weight,
                                       init_method=init_method,
                                       concept_bank=self.concept_bank['MM'],
                                       bias=bias)

    def forward_mm(self, inp, modality):
        inp: dict = self.concept_score(inp)
        inp = self.concept_fusion(inp)
        return self.classifier(inp)


class SALinearCBM(SLinearCBM):
    def __init__(self, concept_bank, idx_to_class=None, n_classes=5, fusion='c', activation='sigmoid',
                 analysis_top_k=5,
                 analysis_threshold=0,
                 init_method='default',
                 modality_mask=False,
                 bias=False,
                 **kwargs
                 ):
        """
        PosthocCBM Linear Layer.
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
            mode (str, optional): Mode of the MM classifier. Defaults to 'c'. ( 'r':residual, 'b': clack)
        """
        super(SALinearCBM, self).__init__(concept_bank=concept_bank, idx_to_class=idx_to_class, n_classes=n_classes,
                                          activation=activation,
                                          analysis_top_k=analysis_top_k,
                                          analysis_threshold=analysis_threshold,
                                          init_method=init_method,
                                          modality_mask=modality_mask,
                                          bias=bias,
                                          **kwargs
                                          )
        # A single linear layer will be used as the classifier
        from models.blocks.layers import SelfLinearAttentionPooling
        self.attn_pool = SelfLinearAttentionPooling(input_dim=self.concept_bank['MM'].n_concepts)

    def pooling_fusion(self, x):
        return self.attn_pool(x)


class MMLinearCBM(CBM):
    def __init__(self, concept_bank, idx_to_class=None, n_classes=5, fusion='max', activation='sigmoid',
                 analysis_top_k=5,
                 analysis_threshold=0,
                 act_on_weight=False,
                 init_method='default',
                 bias=False,
                 **kwargs
                 ):
        """
        PosthocCBM Linear Layer.
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
            mode (str, optional): Mode of the MM classifier. Defaults to 'c'. ( 'r':residual, 'b': clack)
        """
        super(MMLinearCBM, self).__init__(concept_bank,
                                          idx_to_class=idx_to_class,
                                          n_classes=n_classes,
                                          fusion=fusion,
                                          analysis_top_k=analysis_top_k,
                                          analysis_threshold=analysis_threshold,
                                          modality_mask=False,
                                          **kwargs
                                          )
        # A single linear layer will be used as the classifier
        modality_feature_dict = {m: (self.concept_bank[m].n_concepts, self.n_classes) for m in self.mm_order}
        self.classifier = MMWeightMatrix(modality_feature_dict=modality_feature_dict, activation=activation,
                                         act_on_weight=act_on_weight, init_method=init_method, bias=bias)
        self.concept_bank.set_single_modality_score(True)
        self.mode = 'concat'
        print(f'concepts: {[self.concept_bank[m].n_concepts for m in self.mm_order]} '
              f'classes: {self.n_classes} activation: {activation}')

    def forward_mm(self, inp, modality):
        inp: dict = self.concept_score(inp)
        inp = self.concept_fusion(inp)
        return self.classifier[modality](inp)


class M2LinearCBM(CBM):
    def __init__(self, concept_bank, idx_to_class=None, n_classes=5, fusion='max', activation='sigmoid',
                 analysis_top_k=5,
                 analysis_threshold=0,
                 act_on_weight=False,
                 init_method='default',
                 bias=False):
        """
        PosthocCBM Linear Layer.
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
            mode (str, optional): Mode of the MM classifier. Defaults to 'c'. ( 'r':residual, 'b': clack)
        """
        super(M2LinearCBM, self).__init__(concept_bank,
                                          idx_to_class=idx_to_class,
                                          n_classes=n_classes,
                                          fusion=fusion,
                                          analysis_top_k=analysis_top_k,
                                          analysis_threshold=analysis_threshold,
                                          modality_mask=False)
        # A single linear layer will be used as the classifier
        self.modality_map = {'FA': 'FA_ICGA', 'ICGA': 'FA_ICGA', 'US': 'US'}
        self.merged_modalities = ['FA', 'ICGA']

        modality_feature_dict = {m: (self.concept_bank[m].n_concepts, self.n_classes) for m in ['FA_ICGA', 'US']}
        self.classifier = MMWeightMatrix(modality_feature_dict=modality_feature_dict, activation=activation,
                                         act_on_weight=act_on_weight, init_method=init_method,
                                         modality_map=self.modality_map, bias=bias)

        self.concept_bank.set_single_modality_score(True)
        self.concept_bank.set_modality_map(self.modality_map)

        print(f'concepts: {[self.concept_bank[m].n_concepts for m in self.mm_order]} '
              f'classes: {self.n_classes} activation: {activation}')

    def concept_fusion(self, x: dict, mode=None):
        # x (B, T, H)
        x = {k: self.pooling_fusion(v) for k, v in x.items()}
        self.run_hooks(x)
        if all([k in x.keys() for k in self.merged_modalities]):
            merged = torch.stack([x[k] for k in self.merged_modalities], dim=1)
            x[self.merged_modalities[0]] = self.pooling_fusion(merged)
            [x.pop(k) for k in self.merged_modalities[1:]]
        x = torch.concat([x[k] for k in self.mm_order if k in x.keys()], dim=-1)
        return x

    def forward_mm(self, inp, modality):
        inp: dict = self.concept_score(inp)

        inp = self.concept_fusion(inp)
        return self.classifier[modality](inp)
