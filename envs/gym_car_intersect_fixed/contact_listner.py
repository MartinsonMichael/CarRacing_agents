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

        if fixA == 'left_sensor':
            bodyA.left_sensor = True

        if fixB == 'left_sensor':
            bodyB.left_sensor = True

        if sensA and bodyA.name == 'bot_car' and (bodyB.name in {'car', 'bot_car'}):
            if fixB == 'body':
                bodyA.stop = True
        if sensB and bodyB.name == 'bot_car' and (bodyA.name in {'car', 'bot_car'}):
            if fixA == 'body':
                bodyB.stop = True

        # Processing Collision:
        if (bodyA.name in {'car'}) and (bodyB.name in {'car', 'bot_car'}):
            if fixB != 'sensor':
                bodyA.collision = True

        if (bodyA.name in {'car', 'bot_car'}) and (bodyB.name in {'car'}):
            if fixA != 'sensor':
                bodyB.collision = True

        if (bodyA.name in {'car'}) and (bodyB.name in {'car'}):
            if fixA != 'sensor':
                bodyB.collision = True

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

        if sensA and bodyA.name == 'bot_car' and (bodyB.name in {'car', 'bot_car'}):
            if fixB == 'body':
                bodyA.stop = False
        if sensB and bodyB.name == 'bot_car' and (bodyA.name in {'car', 'bot_car'}):
            if fixA == 'body':
                bodyB.stop = False

        # Processing Collision:
        if (bodyA.name in {'car'}) and (bodyB.name in {'car', 'bot_car'}):
            if fixB != 'sensor':
                bodyA.collision = False
        if (bodyA.name in {'car', 'bot_car'}) and (bodyB.name in {'car'}):
            if fixA != 'sensor':
                bodyB.collision = False
