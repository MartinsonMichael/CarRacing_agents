from Box2D.b2 import contactListener


class RefactoredContactListener(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # Data to define sensor data:
        sensA = contact.fixtureA.sensor
        sensB = contact.fixtureB.sensor

        # Data to define collisions:
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData
        # Check data we have for fixtures:
        fixA = contact.fixtureA.userData
        fixB = contact.fixtureB.userData

        if fixA == 'right_sensor':
            bodyA.right_sensor = True

        if fixB == 'right_sensor':
            bodyB.right_sensor = True

        if fixA == 'sensor' and fixB == 'sensor':
            bodyB.collision = True
            bodyA.collision = True

    def EndContact(self, contact):
        sensA = contact.fixtureA.sensor
        sensB = contact.fixtureB.sensor

        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData

        fixA = contact.fixtureA.userData
        fixB = contact.fixtureB.userData

        if fixA == 'right_sensor':
            bodyA.right_sensor = False

        if fixB == 'right_sensor':
            bodyB.right_sensor = False

        if fixA == 'left_sensor':
            bodyA.left_sensor = False

        if fixB == 'left_sensor':
            bodyB.left_sensor = False

        some_set = {'car', 'bot_car', 'wheel', 'body'}
        if bodyA.name in some_set and bodyB.name in some_set:
            if fixA in some_set and fixB in some_set:
                bodyB.collision = False
                bodyA.collision = False
