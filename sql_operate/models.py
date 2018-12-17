from sql_operate.sql import db


class EqxParsed(db.Model):
    """
    爬虫数据表
    """
    __tablename__ = 'eqxiu_scene_parsed'

    id = db.Column(db.VARCHAR, primary_key=True)
    created_time = db.Column(db.VARCHAR, nullable=True)
    name = db.Column(db.VARCHAR, nullable=True)
    view_oss = db.Column(db.VARCHAR, nullable=True)
    ranking = db.Column(db.Integer)
    scene_path = db.Column(db.VARCHAR, nullable=True)
    view_url = db.Column(db.VARCHAR, nullable=True)
    view_size = db.Column(db.VARCHAR, nullable=True)
    price = db.Column(db.VARCHAR, nullable=True)
    color = db.Column(db.VARCHAR, nullable=True)
    heat = db.Column(db.Integer)
    layer = db.Column(db.TEXT)
    tdate = db.Column(db.VARCHAR, nullable=True)
    type = db.Column(db.VARCHAR, nullable=True)
