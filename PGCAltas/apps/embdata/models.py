from django.db import models
from django.core.paginator import Paginator


class EMBManager(models.Manager):

    def all(self):
        return super().filter(is_delete=0).all()

    def logdel(self, **kwargs):
        qs = super().filter(**kwargs)
        qs.is_delete = 1

    def paginator(self, per_row=100, **flt_kwargs):
        qs = super().filter(**flt_kwargs)
        paginator = Paginator(qs, per_row)
        return paginator


class BaseModel(models.Model):

    is_delete = models.BooleanField(default=0, verbose_name='Del')

    class Meta:
        abstract = True

    query = EMBManager()


class GenesInfo(BaseModel):

    access = models.CharField(max_length=20, null=True, verbose_name='AccessNo.')
    name = models.CharField(max_length=30, verbose_name='Name')
    signify = models.BooleanField(default=0, verbose_name='Signify')
    importance = models.FloatField(default=float(0), verbose_name='Importance')

    class Meta:

        db_table = 'emb_genes'
        verbose_name = 'GeneInfo'
        verbose_name_plural = 'GenesInfo'
        ordering = ['importance']


class ClustersInfo(BaseModel):

    name = models.CharField(max_length=30, verbose_name='CellType')

    class Meta:

        db_table = 'emb_clusters'
        verbose_name = 'Cluster'


class StagesInfo(BaseModel):

    stage = models.CharField(max_length=20, verbose_name='Stage')

    class Meta:

        db_table = 'emb_stages'
        verbose_name = 'Stage[day]'
        verbose_name_plural = 'Stages[day]'


class CellsInfo(BaseModel):

    name = models.CharField(max_length=20, verbose_name='CellNo.')
    barcode = models.CharField(max_length=20, verbose_name='BarCode')
    sample = models.SmallIntegerField(verbose_name='SampleNo.')
    type = models.ForeignKey(ClustersInfo, db_constraint=False, db_index=False, related_name='cells')
    stage = models.ForeignKey(StagesInfo, db_constraint=False, db_index=False, related_name='cells')

    class Meta:

        db_table = 'emb_cells'
        verbose_name = 'CellInfo'
        verbose_name_plural = 'CellsInfo'
        ordering = ['stage']

    def __str__(self):
        return "Cell{idx}@{type}_{stage}".format(idx=self.id, type=self.type.name, stage=self.stage.stage)


class Expression(BaseModel):

    expr = models.FloatField(verbose_name='Expression')
    gid = models.ForeignKey(GenesInfo, db_constraint=False, db_index=False, related_name='expr')
    cid = models.ForeignKey(CellsInfo, db_constraint=False, db_index=False, related_name='expr')
    signify = models.BooleanField(default=0, verbose_name='Signify')
    ctype = models.IntegerField(verbose_name='CellType')

    class Meta:

        db_table = 'emb_expr'
        verbose_name = 'Expression'
        verbose_name_plural = verbose_name
        ordering = ['cid_id']

    def __str__(self):
        return "Cell{idx}@{type}_{gene}".format(idx=self.cid_id, type=self.ctype, gene=self.gid_id)
