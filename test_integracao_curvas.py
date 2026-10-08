"""Teste real de fisica e sensores sem depender de janela Pygame."""
import math
import sys
import types
import unittest

# Somente a renderizacao Pygame e substituida no teste.
sys.modules.setdefault('pygame', types.SimpleNamespace())
from car import Car
from curve_safety import CurveSafety
from track import Track

class Hairpin:
    name = 'hairpin_teste'
    spawn_point = (20.0, 55.0)
    spawn_angle = 0.0
    checkpoints = [(100,55), (106,112), (30,135)]
    checkpoint_radius_squared = 18**2
    def is_point_on_track_xy(self,x,y):
        return ((0 <= x <= 120 and 40 <= y <= 70) or
                (90 <= x <= 120 and 40 <= y <= 150) or
                (0 <= x <= 120 and 120 <= y <= 150))
    def checkpoint_reached(self,point,idx):
        x,y=self.checkpoints[idx]
        return (point[0]-x)**2+(point[1]-y)**2 <= self.checkpoint_radius_squared

class Ring:
    name = 'ring_teste'
    spawn_point = (150.0,80.0)
    spawn_angle = 0.0
    checkpoints = [
        (150+70*math.sin(k*2*math.pi/20), 150-70*math.cos(k*2*math.pi/20))
        for k in range(1,21)
    ]
    checkpoint_radius_squared = 18**2
    def is_point_on_track_xy(self,x,y):
        return 50**2 <= (x-150)**2+(y-150)**2 <= 90**2
    def checkpoint_reached(self,point,idx):
        x,y=self.checkpoints[idx]
        return (point[0]-x)**2+(point[1]-y)**2 <= self.checkpoint_radius_squared

class Free:
    spawn_point=(0.0,0.0)
    spawn_angle=0.0
    name = 'free'
    checkpoints=[]
    def is_point_on_track_xy(self,x,y): return True

class DummyBrain:
    def pensar(self, inputs): return {'acelerar':0.6, 'frear':0.0,'virar':0.0}

class DummyEvo:
    def criar_cerebro_inicial(self, pista): return DummyBrain()
    def criar_descendente(self,*args,**kwargs):return DummyBrain()

class SafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Car.base_image=object()
        Car.cache_ready=True
        Car.update_image=lambda self:None

    def test_does_not_intervene_on_open_road(self):
        car=Car(Free(),1,DummyEvo())
        turn, traction, active, _ = CurveSafety.choose(car,Free(), 0.3,0.7)
        self.assertEqual((turn,traction,active),(0.3,0.7,False))

    def test_never_stops(self):
        car=Car(Free(),1,DummyEvo())
        car.brain=type('Braker',(),{'pensar':lambda s,i:{'acelerar':0.,'frear':1.,'virar':0.}})()
        for i in range(120):car.update(1/60,Free())
        self.assertGreaterEqual(car.speed,25)
        self.assertLessEqual(car.speed,175)

    def test_checkpoint_gate_crosses_full_road(self):
        track = Track.__new__(Track)
        # Um portal normal a uma via horizontal, com 18 px de cada lado.
        track.checkpoint_gates = [(100.,55.,1.,0.,0.,1.,18.,18.)]
        self.assertTrue(track.checkpoint_crossed((95.,70.), (106.,70.),0))
        self.assertFalse(track.checkpoint_crossed((95.,90.),(106.,90.),0))
        self.assertFalse(track.checkpoint_crossed((106.,55.),(95.,55.),0))

    def test_closed_curve_can_run_laps_at_speed(self):
        track = Ring()
        car = Car(track,1,DummyEvo())
        car.speed = 80.0
        accumulated_speed=0.0
        for _ in range(60 * 25):
            car.update(1/60,track)
            accumulated_speed += car.speed
        # Mesmo com rede que nunca vira, o sistema deve evitar colisões
        # sucessivas e nao obrigar o carro a andar sempre a 25 px/s.
        self.assertGreaterEqual(car.checkpoints_total,60)
        self.assertLessEqual(car.deaths,2)
        self.assertGreater(accumulated_speed/(60*25),65.0)

    def test_hairpin_collision_prevention_and_progress(self):
        track=Hairpin()
        car=Car(track,2,DummyEvo())
        baseline=CurveSafety.predict(car,track,0.0,1.0)
        # Na primeira decisao, a reta em aceleracao levaria a uma colisao.
        self.assertLess(baseline[0],CurveSafety.HORIZON)
        for tick in range(60*6):
            car.update(1/60,track)
            if car.checkpoints_total >= 2:
                break
        self.assertGreaterEqual(car.checkpoints_total,2,
           'A protecao deveria permitir atravessar duas curvas do teste')
        self.assertEqual(car.deaths,0)
        self.assertGreater(car.safety_interventions,0)

if __name__=='__main__':unittest.main()
